from __future__ import annotations

import json
import random
import re
from typing import Optional, Tuple

from .heu_repair import repair_vmec_decision
from .json_utils import canonicalize_json_obj, parse_json_dict


_COEFF_KEY_PATTERN = re.compile(r"([RZ]B[CS])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)


def _parse_mode_numbers(key: str) -> Tuple[Optional[int], Optional[int]]:
    norm_key = key.strip().replace(" ", "")
    match = _COEFF_KEY_PATTERN.match(norm_key)
    if not match:
        return None, None
    try:
        return abs(int(match.group(2))), abs(int(match.group(3)))
    except Exception:
        return None, None


def _is_high_order_mode(key: str) -> bool:
    m, n = _parse_mode_numbers(key)
    if m is None or n is None:
        return False
    return (m > 2) or (n > 1)


class VmecAdapter:
    def __init__(self, reward_system, config, rng: random.Random):
        self.reward_system = reward_system
        self.config = config
        self.rng = rng

    def gate(self, decision_json: str) -> Optional[str]:
        obj = parse_json_dict(decision_json)
        if obj is None:
            return None
        nc = obj.get("new_coefficients")
        if not isinstance(nc, dict) or not nc:
            return None

        # minimal coercion; real clamp is in HeuRepairOp/RewardingSystem
        cleaned = {}
        for k, v in nc.items():
            if not isinstance(k, str):
                continue
            nk = k.strip().replace(" ", "")
            try:
                fv = float(v)
            except Exception:
                continue
            cleaned[nk] = fv
        if not cleaned:
            return None

        return canonicalize_json_obj({"new_coefficients": cleaned})

    def heu_repair(self, decision_json: str) -> Optional[str]:
        high_order_shrink = float(self.config.get("fusionopt.heu_repair.vmec.high_order_shrink", 0.5))
        max_total_rel = float(self.config.get("fusionopt.heu_repair.vmec.max_total_rel_change", 0.5))
        m_thr = int(self.config.get("fusionopt.heu_repair.vmec.high_order_m_threshold", 2) or 2)
        n_thr = int(self.config.get("fusionopt.heu_repair.vmec.high_order_n_threshold", 1) or 1)
        enable_shrink = bool(self.config.get("fusionopt.heu_repair.vmec.enable_high_order_shrink", True))
        strategy = str(self.config.get("fusionopt.heu_repair.vmec.global_step_strategy", "scale") or "scale")
        metric = str(self.config.get("fusionopt.heu_repair.vmec.global_step_metric", "sum") or "sum")
        return repair_vmec_decision(
            decision_json,
            reward_system=self.reward_system,
            rng=self.rng,
            high_order_shrink=high_order_shrink,
            max_total_rel_change=max_total_rel,
            high_order_m_threshold=m_thr,
            high_order_n_threshold=n_thr,
            enable_high_order_shrink=enable_shrink,
            global_step_strategy=strategy,
            global_step_metric=metric,
        )

    def std_resample(self) -> str:
        base = getattr(self.reward_system, "base_coeffs", {}) or {}
        keys = list(base.keys())
        if not keys:
            return json.dumps({"new_coefficients": {}})

        if bool(self.config.get("fusionopt.std_resample.vmec.high_order_only", False)):
            ho_keys = [k for k in keys if _is_high_order_mode(k)]
            if ho_keys:
                keys = ho_keys

        max_changes = int(self.config.get("llm_constraints.max_coeff_changes", 8) or 8)
        max_changes_cap = int(self.config.get("fusionopt.std_resample.vmec.max_changes_cap", 6) or 6)
        if max_changes_cap <= 0:
            max_changes_cap = 6
        low = float(self.config.get("llm_constraints.low_order_max_rel_change", 0.02) or 0.02)
        high = float(self.config.get("llm_constraints.high_order_max_rel_change", 0.05) or 0.05)

        k = self.rng.randint(1, max(1, min(max_changes, max_changes_cap)))
        sel = self.rng.sample(keys, k)
        new_coeffs = {}
        for kk in sel:
            bv = base.get(kk)
            if bv is None or bv == 0.0:
                continue
            rel = self.rng.uniform(-high, high)
            if abs(rel) < low:
                rel = low if rel >= 0 else -low
            new_coeffs[kk] = float(bv) * (1.0 + rel)

        return json.dumps({"new_coefficients": new_coeffs})

    def std_mutation(self, parent_json: str) -> str:
        obj = parse_json_dict(parent_json) or {}
        nc = obj.get("new_coefficients") if isinstance(obj, dict) else None
        if not isinstance(nc, dict) or not nc:
            return self.std_resample()

        nc2 = dict(nc)
        keys = list(nc2.keys())

        if bool(self.config.get("fusionopt.std_mutation.vmec.high_order_only", False)):
            ho_keys = [k for k in keys if _is_high_order_mode(k)]
            if ho_keys:
                keys = ho_keys

        k = self.rng.randint(1, min(3, len(keys)))
        mutate_keys = self.rng.sample(keys, k)
        max_rel = float(self.config.get("fusionopt.std_mutation.vmec.max_rel_change", 0.05) or 0.05)
        if max_rel <= 0.0:
            max_rel = 0.05
        for kk in mutate_keys:
            try:
                v = float(nc2[kk])
            except Exception:
                continue
            rel = self.rng.uniform(-max_rel, max_rel)
            nc2[kk] = v * (1.0 + rel)

        return json.dumps({"new_coefficients": nc2})

    def std_crossover(self, parent_a_json: str, parent_b_json: str) -> str:
        a = parse_json_dict(parent_a_json) or {}
        b = parse_json_dict(parent_b_json) or {}
        ca = a.get("new_coefficients") if isinstance(a, dict) else None
        cb = b.get("new_coefficients") if isinstance(b, dict) else None
        if not isinstance(ca, dict) or not isinstance(cb, dict):
            return self.std_resample()

        keys = sorted(set(list(ca.keys()) + list(cb.keys())))
        child = {}
        for kk in keys:
            take_a = self.rng.random() < 0.5
            if take_a and kk in ca:
                child[kk] = ca[kk]
            elif (not take_a) and kk in cb:
                child[kk] = cb[kk]
            elif kk in ca:
                child[kk] = ca[kk]
            elif kk in cb:
                child[kk] = cb[kk]

        return json.dumps({"new_coefficients": child})
