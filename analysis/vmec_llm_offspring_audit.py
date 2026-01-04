#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is importable when executing this script via:
#   python analysis/vmec_llm_offspring_audit.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusionopt.adapters_vmec import VmecAdapter
from fusionopt.json_utils import parse_json_dict


@dataclass(frozen=True)
class RunChoice:
    seed: int
    tag: str
    run_dir: Path


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_llm_response_json(response_text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(response_text, str):
        return None
    s = response_text.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    obj = parse_json_dict(s)
    if obj is not None:
        return obj
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        return parse_json_dict(s[i : j + 1])
    return None


def _load_cache_response(cache_dir: Path, cache_key: str) -> Optional[str]:
    if not cache_key:
        return None
    p = cache_dir / f"{cache_key}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    rt = data.get("response_text")
    return str(rt) if isinstance(rt, str) else None


def _llm_candidates_from_logs(run_dir: Path) -> Tuple[Set[str], Dict[str, Any]]:
    logs_path = run_dir / "logs" / "llm_calls.jsonl"
    cache_dir = run_dir / "cache" / "llm"
    calls = _read_jsonl(logs_path)

    by_status: Dict[str, int] = {}
    by_error: Dict[str, int] = {}
    gens: Set[int] = set()
    ok_cache_keys: Set[str] = set()

    for c in calls:
        st = str(c.get("status") or "")
        by_status[st] = int(by_status.get(st, 0)) + 1
        if "generation" in c:
            try:
                gens.add(int(c["generation"]))
            except Exception:
                pass
        err = c.get("error")
        if err is not None and str(err).strip():
            k = f"{st}:{str(err)}"
            by_error[k] = int(by_error.get(k, 0)) + 1

        if st == "ok":
            ck = str(c.get("cache_key") or "").strip()
            if ck:
                ok_cache_keys.add(ck)

    adapter = VmecAdapter(reward_system=None, config=None, rng=None)  # type: ignore[arg-type]
    out: Set[str] = set()

    for ck in sorted(ok_cache_keys):
        rt = _load_cache_response(cache_dir, ck)
        if rt is None:
            continue
        obj = _parse_llm_response_json(rt)
        if obj is None:
            continue
        try:
            gated = adapter.gate(json.dumps(obj))
        except Exception:
            gated = None
        if isinstance(gated, str) and gated:
            out.add(gated)

    meta = {
        "llm_call_count": int(len(calls)),
        "llm_trigger_generations": int(len(gens)),
        "llm_call_status_counts": by_status,
        "llm_call_error_counts": by_error,
        "llm_ok_unique_cache_keys": int(len(ok_cache_keys)),
        "llm_ok_unique_candidates": int(len(out)),
    }
    return out, meta


def _infer_n_obj(df: pd.DataFrame) -> int:
    i = 1
    while f"f{i}_min" in df.columns:
        i += 1
    return i - 1


def _nd_front_mask(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return np.zeros((0,), dtype=bool)
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    mask = np.zeros((F.shape[0],), dtype=bool)
    if nd_idx is None:
        return mask
    for i in list(nd_idx) if isinstance(nd_idx, (list, tuple, np.ndarray)) else []:
        try:
            mask[int(i)] = True
        except Exception:
            pass
    return mask


def audit_run(run_dir: Path) -> Dict[str, Any]:
    df = pd.read_csv(run_dir / "evaluations.csv")
    df = df.copy()
    df["eval_id"] = pd.to_numeric(df.get("eval_id", 0), errors="coerce").fillna(0).astype(int)
    df["feasible"] = pd.to_numeric(df.get("feasible", 0), errors="coerce").fillna(0).astype(int)
    df["status"] = df.get("status", "").astype(str)

    n_obj = _infer_n_obj(df)
    obj_cols = [f"f{i+1}_min" for i in range(n_obj)]
    for c in obj_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    candidates, llm_meta = _llm_candidates_from_logs(run_dir)
    df["is_llm_child"] = df["decision_json"].astype(str).isin(candidates)

    n_rows = int(len(df))
    llm_eval = df[df["is_llm_child"]].copy()

    df_ok = df[(df["feasible"] == 1) & (df["status"] == "ok")].dropna(subset=obj_cols).copy()
    llm_ok = llm_eval[(llm_eval["feasible"] == 1) & (llm_eval["status"] == "ok")].dropna(subset=obj_cols).copy()

    F_ok = df_ok[obj_cols].to_numpy(dtype=float) if not df_ok.empty else np.zeros((0, n_obj), dtype=float)
    nd_mask = _nd_front_mask(F_ok)

    llm_nd_front = 0
    if len(nd_mask) and not df_ok.empty and not llm_ok.empty:
        ok_ids = set(df_ok.loc[nd_mask, "decision_id"].astype(str).tolist())
        llm_nd_front = int(llm_ok["decision_id"].astype(str).isin(ok_ids).sum())

    best = float("-inf")
    total_improve = 0
    llm_total_improve = 0
    eps = 1e-12
    df_sorted = df.sort_values("eval_id", kind="stable")
    for _, r in df_sorted.iterrows():
        if int(r.get("feasible", 0)) != 1 or str(r.get("status", "")) != "ok":
            continue
        t = r.get("total")
        try:
            tv = float(t)
        except Exception:
            continue
        if not np.isfinite(tv):
            continue
        if tv > best + eps:
            total_improve += 1
            if bool(r.get("is_llm_child")):
                llm_total_improve += 1
            best = tv

    status_counts_all = df["status"].value_counts(dropna=False).to_dict()
    status_counts_llm = llm_eval["status"].value_counts(dropna=False).to_dict() if not llm_eval.empty else {}

    out: Dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "n_rows": n_rows,
        "n_obj": int(n_obj),
        "llm": llm_meta,
        "llm_eval_count": int(len(llm_eval)),
        "llm_eval_share": float(len(llm_eval)) / float(n_rows) if n_rows > 0 else 0.0,
        "llm_eval_ok_feasible_count": int(len(llm_ok)),
        "llm_eval_ok_feasible_rate": float(len(llm_ok)) / float(len(llm_eval)) if len(llm_eval) > 0 else 0.0,
        "llm_eval_nd_front_count": int(llm_nd_front),
        "llm_eval_nd_front_rate_among_llm_ok": float(llm_nd_front) / float(len(llm_ok)) if len(llm_ok) > 0 else 0.0,
        "top1_improve_count": int(total_improve),
        "llm_top1_improve_count": int(llm_total_improve),
        "llm_top1_improve_rate_among_improvements": float(llm_total_improve) / float(total_improve) if total_improve > 0 else 0.0,
        "llm_top1_improve_rate_among_llm_eval": float(llm_total_improve) / float(len(llm_eval)) if len(llm_eval) > 0 else 0.0,
        "status_counts_all": status_counts_all,
        "status_counts_llm_eval": status_counts_llm,
    }
    return out


def _run_timestamp(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            ts = data.get("timestamp")
            if isinstance(ts, str) and ts:
                return ts
        except Exception:
            pass
    return run_dir.name


def _choose_latest_completed(candidates: Iterable[Path], max_eval: int) -> Optional[Path]:
    best: Optional[Path] = None
    best_key: Optional[str] = None
    for p in candidates:
        eval_path = p / "evaluations.csv"
        if not eval_path.exists():
            continue
        try:
            df = pd.read_csv(eval_path, usecols=["eval_id"])
            mx = int(pd.to_numeric(df["eval_id"], errors="coerce").fillna(0).max())
        except Exception:
            continue
        if mx < int(max_eval):
            continue
        key = _run_timestamp(p)
        if best is None or (best_key is not None and key > best_key) or best_key is None:
            best = p
            best_key = key
    return best


def find_runs(results_dir: Path, seeds: List[int], max_eval: int) -> List[RunChoice]:
    out: List[RunChoice] = []
    for seed in seeds:
        b_cands = sorted(results_dir.glob(f"Stellarator_VMEC_Phase4HardV6_Baseline_Budget{max_eval}_seed{seed}_*"))
        l_cands = sorted(results_dir.glob(f"Stellarator_VMEC_Phase4HardV6_LLM_SemOp_Budget{max_eval}_OpenAI_seed{seed}_*"))
        b = _choose_latest_completed(b_cands, max_eval=max_eval)
        l = _choose_latest_completed(l_cands, max_eval=max_eval)
        if b is None:
            raise FileNotFoundError(f"No completed baseline run found for seed={seed} under {results_dir}")
        if l is None:
            raise FileNotFoundError(f"No completed llm run found for seed={seed} under {results_dir}")
        out.append(RunChoice(seed=int(seed), tag="baseline", run_dir=b))
        out.append(RunChoice(seed=int(seed), tag="llm", run_dir=l))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results/stellarator_vmec/fusionopt_v1")
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--out-json", type=str, default="analysis_outputs/phase5/audit/vmec_llm_offspring_audit.json")
    args = ap.parse_args()

    results_dir = _resolve_path(args.results_dir)
    max_eval = int(args.max_eval)
    seeds = [int(s) for s in (args.seeds or [])]

    choices = find_runs(results_dir=results_dir, seeds=seeds, max_eval=max_eval)

    per_seed: Dict[str, Any] = {}
    for ch in choices:
        rep = audit_run(ch.run_dir)
        per_seed.setdefault(str(ch.seed), {})[ch.tag] = rep

    out = {
        "problem_id": "stellarator_vmec",
        "max_eval": int(max_eval),
        "seeds": seeds,
        "per_seed": per_seed,
    }

    out_path = _resolve_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
