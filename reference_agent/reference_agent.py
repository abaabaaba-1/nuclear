import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

try:
    from model.LLM import LLM
except Exception:
    LLM = None


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return x / norms


def _extract_json_obj(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _greedy_match_by_cosine(old_dirs: np.ndarray, new_dirs: np.ndarray) -> np.ndarray:
    old_u = _normalize_rows(old_dirs)
    new_u = _normalize_rows(new_dirs)
    sim = old_u @ new_u.T

    k_old = old_u.shape[0]
    k_new = new_u.shape[0]
    if k_old == 0 or k_new == 0:
        return new_u

    used = np.zeros(k_new, dtype=bool)
    matched = np.zeros((k_old, new_u.shape[1]), dtype=float)

    for i in range(k_old):
        scores = sim[i].copy()
        scores[used] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            j = int(np.argmax(sim[i]))
        used[j] = True
        matched[i] = new_u[j]

    return matched


def _farthest_point_sampling(pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pool_u = _normalize_rows(pool)
    n = pool_u.shape[0]
    if n <= k:
        return pool_u[:k]

    idx0 = int(rng.integers(0, n))
    chosen = [idx0]

    min_cos = np.ones(n, dtype=float)
    min_cos = np.minimum(min_cos, pool_u @ pool_u[idx0])

    while len(chosen) < k:
        idx = int(np.argmin(min_cos))
        chosen.append(idx)
        min_cos = np.minimum(min_cos, pool_u @ pool_u[idx])

    return pool_u[chosen]


def _safe_float_list(x: Any, n: int) -> Optional[List[float]]:
    if not isinstance(x, (list, tuple)):
        return None
    if len(x) != n:
        return None
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


class ReferenceAgent:
    def __init__(
        self,
        config,
        n_obj: int,
        pop_size: int,
        seed: int = 42,
        run_dir: Optional[str] = None,
    ):
        self.config = config
        self.n_obj = int(n_obj)
        self.pop_size = int(pop_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.enabled = bool(self.config.get("reference_agent.enabled", False))
        self.mode = str(self.config.get("reference_agent.mode", "heuristic")).lower().strip()

        self.update_every = int(self.config.get("reference_agent.update_every", 10))
        self.warmup_evals = int(self.config.get("reference_agent.warmup_evals", 0))
        self.min_nd_points = int(self.config.get("reference_agent.min_nd_points", max(2, self.n_obj)))
        self.max_nd_pool = int(self.config.get("reference_agent.max_nd_pool", 512))

        self.base_blend = float(self.config.get("reference_agent.blend", 0.2))
        self.max_angle_change_rad = float(self.config.get("reference_agent.max_angle_change_rad", 0.35))

        self.pool_fallback = str(self.config.get("reference_agent.pool_fallback", "feasible")).lower().strip()
        self.log_attempts = bool(self.config.get("reference_agent.log_attempts", True))

        self.save_ref_dirs = bool(self.config.get("reference_agent.save_ref_dirs", True))
        self.run_dir = run_dir
        self.log_path = None
        if self.run_dir:
            self.log_path = os.path.join(self.run_dir, "reference_agent_log.jsonl")

        self.last_update_generation: Optional[int] = None
        self._initial_logged = False
        self.llm = None
        self._llm_last_error: Optional[str] = None
        if self.enabled and self.mode == "llm" and LLM is not None:
            model_name = self.config.get("model.name", "chatgpt")
            try:
                self.llm = LLM(model=model_name, config=self.config)
            except Exception:
                self.llm = None

    def log_initial(self, ref_dirs: np.ndarray, generation: int = 0, eval_idx: int = 0, eval_budget: int = 0):
        if self._initial_logged:
            return
        if not self.enabled:
            return
        if not self.run_dir:
            return

        record = {
            "event": "init",
            "generation": int(generation),
            "eval_idx": int(eval_idx),
            "eval_budget": int(eval_budget),
            "mode": self.mode,
            "pop_size": int(self.pop_size),
            "n_obj": int(self.n_obj),
        }
        self._log_update(record, generation=int(generation), ref_dirs=_normalize_rows(np.asarray(ref_dirs, dtype=float)))
        self._initial_logged = True

    def _collect_feasible_F(self, items: List[Any]) -> np.ndarray:
        rows = []
        for it in items:
            scores = getattr(it, "scores", None)
            if scores is None:
                continue
            ok = True
            rec = getattr(it, "eval_record", None)
            if rec is not None:
                try:
                    ok = int(getattr(rec, "feasible", 1)) == 1
                except Exception:
                    ok = True
            if not ok:
                continue
            try:
                row = np.asarray(scores, dtype=float)
            except Exception:
                continue
            if row.shape[0] != self.n_obj:
                continue
            if not np.all(np.isfinite(row)):
                continue
            rows.append(row)

        if not rows:
            return np.zeros((0, self.n_obj), dtype=float)
        return np.vstack(rows)

    def _build_observation(self, F: np.ndarray, F_nd: np.ndarray, ref_dirs: np.ndarray, generation: int, eval_idx: int, eval_budget: int) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "n_obj": int(self.n_obj),
            "pop_size": int(self.pop_size),
            "generation": int(generation),
            "eval_idx": int(eval_idx),
            "eval_budget": int(eval_budget),
            "n_feasible": int(F.shape[0]),
            "n_nd": int(F_nd.shape[0]),
        }

        z_min = F.min(axis=0) if F.size else np.zeros(self.n_obj)
        z_max = F.max(axis=0) if F.size else np.ones(self.n_obj)

        obs["objective_min"] = [float(x) for x in z_min.tolist()]
        obs["objective_max"] = [float(x) for x in z_max.tolist()]

        occ = self._compute_occupancy(ref_dirs, F_nd)
        if occ is not None and occ.size:
            obs["occupancy_zero_frac"] = float(np.mean(occ == 0))
            obs["occupancy_min"] = int(np.min(occ))
            obs["occupancy_max"] = int(np.max(occ))
            obs["occupancy_mean"] = float(np.mean(occ))
            if int(occ.size) <= 64:
                obs["occupancy"] = [int(x) for x in occ.tolist()]
        return obs

    def _compute_occupancy(self, ref_dirs: np.ndarray, F_nd: np.ndarray) -> Optional[np.ndarray]:
        if ref_dirs is None or F_nd is None:
            return None
        if ref_dirs.size == 0 or F_nd.size == 0:
            return None

        ref_u = _normalize_rows(np.asarray(ref_dirs, dtype=float))

        z_min = F_nd.min(axis=0)
        z_max = F_nd.max(axis=0)
        denom = np.where(z_max - z_min < 1e-12, 1.0, z_max - z_min)
        F_norm = (F_nd - z_min) / denom
        f_u = _normalize_rows(np.maximum(F_norm, 0.0))

        cos = np.clip(f_u @ ref_u.T, -1.0, 1.0)
        k_idx = np.argmax(cos, axis=1)
        occ = np.zeros(ref_u.shape[0], dtype=int)
        for j in k_idx:
            occ[int(j)] += 1
        return occ

    def _llm_policy(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None

        prompt = (
            "You are an LLM macro-agent that adapts reference vectors for RVEA/K-RVEA (minimization).\n"
            "Given an observation JSON, propose how to update reference vectors.\n"
            "Return JSON only with keys: action, preference, strength.\n"
            "- action: 'keep' | 'bias' | 'resample'\n"
            "- preference: list of length n_obj, positive numbers summing to 1 (only if action='bias')\n"
            "- strength: float in [0,1], 0=very small change, 1=larger change\n"
            "Observation JSON:\n"
            + json.dumps(obs, ensure_ascii=False)
        )

        try:
            resp = self.llm.chat(prompt)
        except Exception as e:
            self._llm_last_error = str(e)
            return None

        data = _extract_json_obj(resp)
        if not isinstance(data, dict):
            return None

        action = str(data.get("action", "")).lower().strip()
        if action not in {"keep", "bias", "resample"}:
            return None

        strength = data.get("strength", 0.0)
        try:
            strength = float(strength)
        except Exception:
            strength = 0.0
        strength = float(np.clip(strength, 0.0, 1.0))

        pref = _safe_float_list(data.get("preference"), self.n_obj)
        if action == "bias":
            if pref is None:
                return None
            pref_arr = np.asarray(pref, dtype=float)
            pref_arr = np.clip(pref_arr, 1e-9, None)
            pref_arr = pref_arr / float(np.sum(pref_arr))
            pref = [float(x) for x in pref_arr.tolist()]

        if action == "resample":
            pref = None

        return {"action": action, "strength": strength, "preference": pref}

    def _propose_ref_dirs_from_nd(self, F_nd: np.ndarray, preference: Optional[List[float]] = None, strength: float = 1.0) -> Optional[np.ndarray]:
        if F_nd is None or F_nd.size == 0:
            return None

        z_min = F_nd.min(axis=0)
        z_max = F_nd.max(axis=0)
        denom = np.where(z_max - z_min < 1e-12, 1.0, z_max - z_min)
        X = (F_nd - z_min) / denom
        X = np.maximum(X, 0.0)

        norms = np.linalg.norm(X, axis=1)
        mask = norms > 1e-12
        X = X[mask]
        if X.size == 0:
            return None

        if X.shape[0] > self.max_nd_pool:
            idx = self.rng.choice(X.shape[0], size=self.max_nd_pool, replace=False)
            X = X[idx]

        pool = [_normalize_rows(X)]

        if preference is not None and strength > 0.0:
            pref = np.asarray(preference, dtype=float)
            pref = np.clip(pref, 1e-9, None)
            pref = pref / float(np.sum(pref))
            concentration = float(2.0 + 18.0 * float(np.clip(strength, 0.0, 1.0)))
            alpha = pref * concentration
            try:
                bias = self.rng.dirichlet(alpha, size=max(int(self.pop_size), 1))
                pool.append(np.asarray(bias, dtype=float))
            except Exception:
                pass

        pool_mat = np.vstack(pool)
        pool_mat = np.maximum(pool_mat, 0.0)
        pool_mat = _normalize_rows(pool_mat)
        if pool_mat.shape[0] < self.pop_size:
            extra = self.rng.dirichlet(np.ones(self.n_obj), size=self.pop_size - pool_mat.shape[0])
            pool_mat = np.vstack([pool_mat, np.asarray(extra, dtype=float)])
            pool_mat = _normalize_rows(pool_mat)

        return _farthest_point_sampling(pool_mat, self.pop_size, self.rng)

    def _apply_safe_update(self, old_dirs: np.ndarray, proposed_dirs: np.ndarray, blend: float) -> np.ndarray:
        old_u = _normalize_rows(np.asarray(old_dirs, dtype=float))
        prop_u = _normalize_rows(np.asarray(proposed_dirs, dtype=float))

        if old_u.shape != prop_u.shape:
            return old_u

        matched = _greedy_match_by_cosine(old_u, prop_u)

        blend = float(np.clip(blend, 0.0, 1.0))
        updated = (1.0 - blend) * old_u + blend * matched
        updated = np.maximum(updated, 0.0)
        updated = _normalize_rows(updated)

        if self.max_angle_change_rad > 0.0:
            cos = np.sum(old_u * updated, axis=1)
            cos = np.clip(cos, -1.0, 1.0)
            angles = np.arccos(cos)
            mask = angles > float(self.max_angle_change_rad)
            if np.any(mask):
                t = float(self.max_angle_change_rad)
                a = angles[mask]
                ratio = (t / (a + 1e-12)).reshape(-1, 1)
                limited = (1.0 - ratio) * old_u[mask] + ratio * updated[mask]
                limited = _normalize_rows(np.maximum(limited, 0.0))
                updated[mask] = limited

        return updated

    def _log_update(self, record: Dict[str, Any], generation: int, ref_dirs: np.ndarray):
        if not self.run_dir:
            return

        if self.save_ref_dirs:
            try:
                np.save(os.path.join(self.run_dir, f"ref_dirs_gen_{int(generation):04d}.npy"), ref_dirs)
            except Exception:
                pass

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def maybe_update(self, ref_dirs: np.ndarray, archive_items: List[Any], generation: int, eval_idx: int, eval_budget: int) -> np.ndarray:
        if not self.enabled:
            return ref_dirs
        if self.update_every <= 0:
            return ref_dirs
        if eval_idx < self.warmup_evals:
            return ref_dirs

        if self.last_update_generation is not None:
            if int(generation) - int(self.last_update_generation) < int(self.update_every):
                return ref_dirs

        F = self._collect_feasible_F(archive_items)
        if F.shape[0] < max(int(self.min_nd_points), 1):
            if self.log_attempts:
                self._log_update(
                    {
                        "event": "skip",
                        "reason": "too_few_feasible",
                        "generation": int(generation),
                        "eval_idx": int(eval_idx),
                        "eval_budget": int(eval_budget),
                        "mode": self.mode,
                        "n_feasible": int(F.shape[0]),
                    },
                    generation=int(generation),
                    ref_dirs=_normalize_rows(np.asarray(ref_dirs, dtype=float)),
                )
            self.last_update_generation = int(generation)
            return ref_dirs

        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F_nd = F[nd_idx]
        pool_points = F_nd
        if pool_points.shape[0] < max(int(self.min_nd_points), 1):
            if self.pool_fallback in {"feasible", "all"}:
                pool_points = F
            else:
                if self.log_attempts:
                    self._log_update(
                        {
                            "event": "skip",
                            "reason": "too_few_nd",
                            "generation": int(generation),
                            "eval_idx": int(eval_idx),
                            "eval_budget": int(eval_budget),
                            "mode": self.mode,
                            "n_feasible": int(F.shape[0]),
                            "n_nd": int(F_nd.shape[0]),
                        },
                        generation=int(generation),
                        ref_dirs=_normalize_rows(np.asarray(ref_dirs, dtype=float)),
                    )
                self.last_update_generation = int(generation)
                return ref_dirs

        obs = self._build_observation(F, F_nd, ref_dirs, generation, eval_idx, eval_budget)

        policy = None
        if self.mode == "llm":
            policy = self._llm_policy(obs)

            if policy is None and self._llm_last_error and self.log_attempts:
                self._log_update(
                    {
                        "event": "llm_error",
                        "generation": int(generation),
                        "eval_idx": int(eval_idx),
                        "eval_budget": int(eval_budget),
                        "mode": self.mode,
                        "error": str(self._llm_last_error)[:500],
                        "obs": obs,
                    },
                    generation=int(generation),
                    ref_dirs=_normalize_rows(np.asarray(ref_dirs, dtype=float)),
                )

        if policy is None:
            policy = {"action": "bias", "strength": 1.0, "preference": None}

        action = str(policy.get("action", "bias")).lower().strip()
        strength = float(policy.get("strength", 1.0) or 0.0)
        strength = float(np.clip(strength, 0.0, 1.0))
        preference = policy.get("preference")

        if action == "keep" or strength <= 0.0:
            if self.log_attempts:
                self._log_update(
                    {
                        "event": "keep",
                        "generation": int(generation),
                        "eval_idx": int(eval_idx),
                        "eval_budget": int(eval_budget),
                        "mode": self.mode,
                        "action": str(action),
                        "strength": float(strength),
                        "n_obj": int(self.n_obj),
                    },
                    generation=int(generation),
                    ref_dirs=_normalize_rows(np.asarray(ref_dirs, dtype=float)),
                )
            self.last_update_generation = int(generation)
            return ref_dirs

        if action == "resample":
            try:
                proposed = self.rng.dirichlet(np.ones(self.n_obj), size=max(int(self.pop_size), 1))
                proposed = _normalize_rows(np.asarray(proposed, dtype=float))
            except Exception:
                proposed = None
        else:
            proposed = self._propose_ref_dirs_from_nd(pool_points, preference=preference, strength=strength)
        if proposed is None:
            return ref_dirs

        blend = float(np.clip(self.base_blend * strength, 0.0, 1.0))
        updated = self._apply_safe_update(ref_dirs, proposed, blend=blend)

        record = {
            "generation": int(generation),
            "eval_idx": int(eval_idx),
            "eval_budget": int(eval_budget),
            "mode": self.mode,
            "action": action,
            "strength": float(strength),
            "blend": float(blend),
            "preference": preference,
            "obs": obs,
        }
        self._log_update(record, generation=int(generation), ref_dirs=updated)

        self.last_update_generation = int(generation)
        return updated
