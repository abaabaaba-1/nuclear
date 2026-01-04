from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

import httpx

from .json_utils import parse_json_dict


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _sha256_hex(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _truncate(s: str, max_len: int = 2048) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def make_cache_key(prompt: str, provider: str, base_url: str, model: str, temperature: float, schema_version: str) -> str:
    payload = f"{prompt}\n{provider}\n{base_url}\n{model}\n{temperature}\n{schema_version}"
    return _sha256_hex(payload)


def _normalize_base_url(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    return s.rstrip("/")


def _read_api_key_file(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        return ""
    p = os.path.expanduser(path.strip())
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = str(line).strip()
                if s:
                    return s
    except Exception:
        return ""
    return ""


@dataclass
class SemOpCandidate:
    decision_json: str
    meta: Dict


class LlmCallsLogger:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "logs", "llm_calls.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()

    def log(self, entry: Dict) -> None:
        entry2 = dict(entry or {})
        entry2.setdefault("ts", _utc_ts())
        line = json.dumps(entry2, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


class LlmClient:
    def __init__(self, provider: str, model: str, api_key: str, base_url: str = "", timeout_sec: float = 60.0):
        self.provider = str(provider or "").strip().lower() or "gemini"
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "")
        self.base_url = _normalize_base_url(str(base_url or ""))
        self.timeout_sec = float(timeout_sec)

    def generate_text(self, prompt: str, temperature: float) -> str:
        if self.provider == "openai":
            return self._generate_text_openai(prompt, temperature)
        return self._generate_text_gemini(prompt, temperature)

    def _generate_text_openai(self, prompt: str, temperature: float) -> str:
        base = self.base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                resp = getattr(e, "response", None)
                code = int(getattr(resp, "status_code", 0) or 0)
                txt = ""
                try:
                    txt = resp.text if resp is not None else ""
                except Exception:
                    txt = ""
                lowered = str(txt).lower()
                if code in (400, 422) and ("response_format" in lowered or "unknown" in lowered or "unsupported" in lowered):
                    payload2 = dict(payload)
                    payload2.pop("response_format", None)
                    r2 = client.post(url, headers=headers, json=payload2)
                    r2.raise_for_status()
                    data = r2.json()
                else:
                    raise
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            return json.dumps(data)

    def _generate_text_gemini(self, prompt: str, temperature: float) -> str:
        model = self.model or "gemini-2.5-pro"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature)},
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            r = client.post(url, params=params, json=payload)
            r.raise_for_status()
            data = r.json()
        try:
            c0 = data.get("candidates", [])[0]
            parts = (c0.get("content") or {}).get("parts") or []
            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                return str(parts[0]["text"])
        except Exception:
            pass
        return json.dumps(data)


class LlmSemOpManager:
    def __init__(self, config, run_dir: str, problem_id: str, adapter, seed: int):
        self.config = config
        self.run_dir = run_dir
        self.problem_id = str(problem_id)
        self.adapter = adapter
        self.seed = int(seed)

        self.enabled = bool(config.get("fusionopt.llm_semop.enabled", False))
        self.trigger_type = str(config.get("fusionopt.llm_semop.trigger.type", "fixed_interval") or "fixed_interval")
        self.every_n_generations = int(config.get("fusionopt.llm_semop.trigger.every_n_generations", 5) or 5)
        self.batch_size = int(config.get("fusionopt.llm_semop.batch_size", 8) or 8)

        self.schema_version = str(config.get("fusionopt.llm_semop.schema_version", "v1") or "v1")

        temp = float(config.get("fusionopt.llm_semop.temperature", 0.0) or 0.0)
        self.temperature = 0.0 if abs(temp) > 1e-12 else 0.0

        self.model_name = str(config.get("model.name", "") or "")
        self.api_key = str(config.get("model.api_key", "") or "")
        self.provider = str(config.get("model.provider", "") or "")
        self.base_url = str(config.get("model.base_url", "") or "")

        if not self.api_key:
            env_key = (
                os.environ.get("FUSIONOPT_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if env_key:
                self.api_key = str(env_key)

        if not self.api_key:
            key_file = str(config.get("model.api_key_file", "") or "")
            self.api_key = _read_api_key_file(key_file)

        self._logger = LlmCallsLogger(run_dir)

        self._cache_dir = os.path.join(run_dir, "cache", "llm")
        self._queue: Deque[SemOpCandidate] = deque()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._future: Optional[Future] = None
        self._last_trigger_generation: Optional[int] = None
        self._lock = threading.Lock()

        if self.enabled:
            os.makedirs(self._cache_dir, exist_ok=True)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_semop")

    def close(self) -> None:
        with self._lock:
            ex = self._executor
            self._executor = None
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _should_trigger(self, generation: int) -> bool:
        if not self.enabled:
            return False
        if str(self.trigger_type).strip().lower() != "fixed_interval":
            return False
        if self.every_n_generations <= 0:
            return False
        return int(generation) % int(self.every_n_generations) == 0

    def _drain_future_if_done(self) -> None:
        fut = self._future
        if fut is None:
            return
        if not fut.done():
            return
        self._future = None
        try:
            res = fut.result() or []
        except Exception:
            return
        for c in res:
            if isinstance(c, SemOpCandidate) and isinstance(c.decision_json, str) and c.decision_json:
                self._queue.append(c)

    def maybe_schedule_batch(self, generation: int, population) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._drain_future_if_done()
            if self._future is not None:
                return
            if not self._should_trigger(generation):
                return
            if self._last_trigger_generation == int(generation):
                return
            self._last_trigger_generation = int(generation)

            prompts = self._build_batch_prompts(generation, population)
            ex = self._executor
            if ex is None:
                return
            self._future = ex.submit(self._worker_generate_batch, int(generation), prompts)

    def try_get_candidate(self) -> Optional[SemOpCandidate]:
        if not self.enabled:
            return None
        with self._lock:
            self._drain_future_if_done()
            if self._queue:
                return self._queue.popleft()
        return None

    def log_validation_failed(self, generation: int, meta: Dict, error: str) -> None:
        cache_key = (meta or {}).get("cache_key")
        cache_hit = bool((meta or {}).get("cache_hit", False))
        prompt_hash = (meta or {}).get("prompt_hash")
        self._logger.log(
            {
                "generation": int(generation),
                "operator": "llm_semop",
                "provider": self.provider or "gemini",
                "model": self.model_name,
                "temperature": float(self.temperature),
                "schema_version": self.schema_version,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "status": "validation_failed",
                "error": str(error or ""),
                "prompt_hash": prompt_hash,
            }
        )

    def _build_batch_prompts(self, generation: int, population) -> List[str]:
        parent_jsons: List[str] = []
        for p in list(population or []):
            try:
                parent_jsons.append(str(getattr(p, "value", "") or ""))
            except Exception:
                parent_jsons.append("")

        if not parent_jsons:
            parent_jsons = ["{}"]

        rng = __import__("random").Random(int(self.seed) + int(generation) * 1000003)

        prompts = []
        for _ in range(max(0, int(self.batch_size))):
            pj = parent_jsons[rng.randint(0, len(parent_jsons) - 1)]
            prompts.append(self._make_prompt(parent_decision_json=pj))
        return prompts

    def _make_prompt(self, parent_decision_json: str) -> str:
        if self.problem_id == "stellarator_vmec":
            max_changes = 12
            try:
                max_changes = int(getattr(self, "config", None).get("llm_constraints.max_coeff_changes", max_changes))
            except Exception:
                max_changes = 12
            max_changes = max(1, int(max_changes))

            schema = '{"new_coefficients": {"RBC(1,1)": -0.236, "ZBS(2,2)": -0.054}}'
            return (
                "You are a semantic editing operator for a multi-objective optimizer.\n"
                "Return ONLY a valid JSON object. No markdown, no code fences, no extra keys.\n"
                f"Schema: {schema}\n"
                f"Rules: new_coefficients must be a non-empty object with 1..{max_changes} entries.\n"
                "Rules: Only edit keys that already exist in the parent decision_json new_coefficients.\n"
                "Rules: Use float values. Keep changes small (avoid extreme magnitudes).\n"
                f"Parent decision_json: {parent_decision_json}\n"
            )

        max_cell_changes = 10
        try:
            max_cell_changes = int(getattr(self, "config", None).get("llm_constraints.max_cell_changes", max_cell_changes))
        except Exception:
            max_cell_changes = 10
        max_cell_changes = max(1, int(max_cell_changes))

        schema = '{"cells": [[0, 0, 1], [1, 2, -1]]}'
        return (
            "You are a semantic editing operator for a multi-objective optimizer.\n"
            "Return ONLY a valid JSON object. No markdown, no code fences, no extra keys.\n"
            f"Schema: {schema}\n"
            f"Rules: cells must be a list with at most {max_cell_changes} entries.\n"
            "Rules: each entry is [phi:int, theta:int, state:int] where state is -1 or 1 (no 0).\n"
            f"Parent decision_json: {parent_decision_json}\n"
        )

    def _cache_path(self, cache_key: str) -> str:
        return os.path.join(self._cache_dir, f"{cache_key}.json")

    def _read_cache(self, cache_key: str) -> Optional[Dict]:
        p = self._cache_path(cache_key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, cache_key: str, payload: Dict) -> None:
        p = self._cache_path(cache_key)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _worker_generate_batch(self, generation: int, prompts: List[str]) -> List[SemOpCandidate]:
        out: List[SemOpCandidate] = []
        client = LlmClient(provider=self.provider, model=self.model_name, api_key=self.api_key, base_url=self.base_url)

        for prompt in list(prompts or []):
            prompt = str(prompt or "")
            prompt_hash = _sha256_hex(prompt)
            cache_key = make_cache_key(prompt, self.provider or "", self.base_url or "", self.model_name, self.temperature, self.schema_version)

            t0 = time.time()
            cache_hit = False
            response_text = ""

            cached = self._read_cache(cache_key)
            if isinstance(cached, dict) and "response_text" in cached:
                cache_hit = True
                response_text = str(cached.get("response_text") or "")
            else:
                if not self.api_key:
                    self._logger.log(
                        {
                            "generation": int(generation),
                            "operator": "llm_semop",
                            "provider": self.provider or "gemini",
                            "base_url": self.base_url or "",
                            "model": self.model_name,
                            "temperature": float(self.temperature),
                            "schema_version": self.schema_version,
                            "cache_key": cache_key,
                            "cache_hit": False,
                            "status": "skipped",
                            "error": "missing_api_key",
                            "prompt_hash": prompt_hash,
                            "prompt_preview": _truncate(prompt),
                        }
                    )
                    continue

                try:
                    response_text = client.generate_text(prompt, temperature=self.temperature)
                    self._write_cache(
                        cache_key,
                        {
                            "model": self.model_name,
                            "provider": self.provider or "gemini",
                            "base_url": self.base_url or "",
                            "temperature": float(self.temperature),
                            "schema_version": self.schema_version,
                            "prompt_hash": prompt_hash,
                            "response_text": response_text,
                        },
                    )
                except Exception as e:
                    dt = time.time() - t0
                    self._logger.log(
                        {
                            "generation": int(generation),
                            "operator": "llm_semop",
                            "provider": self.provider or "gemini",
                            "base_url": self.base_url or "",
                            "model": self.model_name,
                            "temperature": float(self.temperature),
                            "schema_version": self.schema_version,
                            "cache_key": cache_key,
                            "cache_hit": False,
                            "status": "error",
                            "latency_sec": float(dt),
                            "error": str(e),
                            "prompt_hash": prompt_hash,
                            "prompt_preview": _truncate(prompt),
                        }
                    )
                    continue

            dt = time.time() - t0

            obj = self._parse_response_json(response_text)
            if obj is None:
                self._logger.log(
                    {
                        "generation": int(generation),
                        "operator": "llm_semop",
                        "provider": self.provider or "gemini",
                        "base_url": self.base_url or "",
                        "model": self.model_name,
                        "temperature": float(self.temperature),
                        "schema_version": self.schema_version,
                        "cache_key": cache_key,
                        "cache_hit": bool(cache_hit),
                        "status": "invalid_json",
                        "latency_sec": float(dt),
                        "error": "not_a_json_object",
                        "prompt_hash": prompt_hash,
                        "prompt_preview": _truncate(prompt),
                        "response_preview": _truncate(response_text),
                    }
                )
                continue

            if not self._validate_schema(obj):
                self._logger.log(
                    {
                        "generation": int(generation),
                        "operator": "llm_semop",
                        "provider": self.provider or "gemini",
                        "base_url": self.base_url or "",
                        "model": self.model_name,
                        "temperature": float(self.temperature),
                        "schema_version": self.schema_version,
                        "cache_key": cache_key,
                        "cache_hit": bool(cache_hit),
                        "status": "validation_failed",
                        "latency_sec": float(dt),
                        "error": "schema_mismatch",
                        "prompt_hash": prompt_hash,
                        "prompt_preview": _truncate(prompt),
                        "response_preview": _truncate(response_text),
                    }
                )
                continue

            gated = self.adapter.gate(json.dumps(obj))
            if gated is None:
                self._logger.log(
                    {
                        "generation": int(generation),
                        "operator": "llm_semop",
                        "provider": self.provider or "gemini",
                        "base_url": self.base_url or "",
                        "model": self.model_name,
                        "temperature": float(self.temperature),
                        "schema_version": self.schema_version,
                        "cache_key": cache_key,
                        "cache_hit": bool(cache_hit),
                        "status": "validation_failed",
                        "latency_sec": float(dt),
                        "error": "adapter_gate_failed",
                        "prompt_hash": prompt_hash,
                        "prompt_preview": _truncate(prompt),
                        "response_preview": _truncate(response_text),
                    }
                )
                continue

            self._logger.log(
                {
                    "generation": int(generation),
                    "operator": "llm_semop",
                    "provider": self.provider or "gemini",
                    "base_url": self.base_url or "",
                    "model": self.model_name,
                    "temperature": float(self.temperature),
                    "schema_version": self.schema_version,
                    "cache_key": cache_key,
                    "cache_hit": bool(cache_hit),
                    "status": "ok",
                    "latency_sec": float(dt),
                    "prompt_hash": prompt_hash,
                }
            )

            out.append(
                SemOpCandidate(
                    decision_json=gated,
                    meta={
                        "cache_key": cache_key,
                        "cache_hit": bool(cache_hit),
                        "prompt_hash": prompt_hash,
                    },
                )
            )

        return out

    def _parse_response_json(self, response_text: str) -> Optional[Dict]:
        if not isinstance(response_text, str):
            return None
        s = response_text.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.strip()
        obj = parse_json_dict(s)
        if obj is not None:
            return obj
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            return parse_json_dict(s[i : j + 1])
        return None

    def _validate_schema(self, obj: Dict) -> bool:
        if not isinstance(obj, dict):
            return False

        if self.problem_id == "stellarator_vmec":
            nc = obj.get("new_coefficients")
            if not isinstance(nc, dict) or not nc:
                return False
            for k, v in nc.items():
                if not isinstance(k, str):
                    return False
                if not isinstance(v, (int, float)):
                    try:
                        float(v)
                    except Exception:
                        return False
            return True

        cells = obj.get("cells")
        if not isinstance(cells, list):
            return False
        for c in cells:
            if not isinstance(c, (list, tuple)) or len(c) != 3:
                return False
            try:
                int(c[0])
                int(c[1])
                int(c[2])
            except Exception:
                return False
        return True
