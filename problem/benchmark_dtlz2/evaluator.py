import json
import random
from typing import Dict, List

import numpy as np
from pymoo.problems.many.dtlz import DTLZ2


def _as_int(val, default: int) -> int:
    try:
        v = int(val)
        return v
    except Exception:
        return int(default)


def _coeff_keys(n_var: int) -> List[str]:
    return [f"x{i}" for i in range(int(n_var))]


def _build_base_coeffs(n_var: int, base_value: float = 0.0) -> Dict[str, float]:
    return {k: float(base_value) for k in _coeff_keys(int(n_var))}


def _decision_to_x(decision_json: str, base_coeffs: Dict[str, float], n_var: int) -> np.ndarray:
    payload = None
    try:
        payload = json.loads(decision_json)
    except Exception:
        payload = None

    delta = {}
    if isinstance(payload, dict):
        delta = payload.get("new_coefficients", {}) or {}

    x = np.zeros(int(n_var), dtype=float)
    for i, k in enumerate(_coeff_keys(int(n_var))):
        v = delta.get(k, base_coeffs.get(k, 0.0))
        try:
            x[i] = float(v)
        except Exception:
            x[i] = float(base_coeffs.get(k, 0.0))

    x = np.clip(x, 0.0, 1.0)
    return x


def generate_initial_population(config, seed: int) -> List[str]:
    random.seed(int(seed))
    np.random.seed(int(seed))

    pop_size = _as_int(config.get("optimization.pop_size", 50), 50)
    n_var = _as_int(config.get("benchmark.n_var", 12), 12)

    jsons: List[str] = []
    keys = _coeff_keys(n_var)

    for _ in range(int(pop_size)):
        coeffs = {k: float(np.random.rand()) for k in keys}
        jsons.append(json.dumps({"new_coefficients": coeffs}))

    return jsons


def _mutate_seed_coefficients(seed_coeffs: Dict[str, float]) -> Dict[str, float]:
    mutated = dict(seed_coeffs or {})
    keys = list(mutated.keys())
    if not keys:
        return mutated

    for key in keys:
        mutated[key] = float(random.random())
    return mutated


def _extract_delta_coefficients(base_coeffs: Dict[str, float], mutated_coeffs: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    base_coeffs = base_coeffs or {}
    mutated_coeffs = mutated_coeffs or {}
    for key, mutated_val in mutated_coeffs.items():
        base_val = base_coeffs.get(key)
        if base_val is None:
            deltas[key] = float(mutated_val)
            continue
        try:
            mv = float(mutated_val)
        except Exception:
            continue
        if abs(mv - float(base_val)) > float(eps):
            deltas[key] = mv
    return deltas


class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.objs = list(config.get("goals") or [])
        self.n_obj = int(len(self.objs))
        self.n_var = _as_int(config.get("benchmark.n_var", 12), 12)

        self.base_coeffs = _build_base_coeffs(self.n_var, base_value=0.0)
        self.problem = DTLZ2(n_var=int(self.n_var), n_obj=int(self.n_obj))

    def evaluate(self, items):
        if not items:
            return [], {}

        X = np.vstack([_decision_to_x(it.value, self.base_coeffs, self.n_var) for it in items])
        F = self.problem.evaluate(X)

        for idx, it in enumerate(items):
            f = np.asarray(F[idx], dtype=float).reshape(-1)
            original_results = {obj: float(f[j]) for j, obj in enumerate(self.objs)}
            transformed_results = dict(original_results)

            results = {
                "original_results": original_results,
                "transformed_results": transformed_results,
                "overall_score": float(-np.sum(f)),
            }
            it.assign_results(results)

        return items, {}
