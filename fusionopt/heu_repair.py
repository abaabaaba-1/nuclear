from __future__ import annotations

import json
import math
import random
import re
from typing import Dict, Optional, Tuple

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


def repair_vmec_decision(
    decision_json: str,
    reward_system,
    rng: random.Random,
    high_order_shrink: float,
    max_total_rel_change: float,
    high_order_m_threshold: int = 2,
    high_order_n_threshold: int = 1,
    enable_high_order_shrink: bool = True,
    global_step_strategy: str = "scale",
    global_step_metric: str = "sum",
) -> Optional[str]:
    obj = parse_json_dict(decision_json)
    if obj is None:
        return None

    new_coeffs = obj.get("new_coefficients")
    if not isinstance(new_coeffs, dict) or not new_coeffs:
        return None

    # 1) key normalization + float coercion
    cleaned: Dict[str, float] = {}
    for k, v in new_coeffs.items():
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

    # 2) use evaluator's clamp (step limits / max_coeff_changes)
    try:
        cleaned = reward_system._sanitize_new_coefficients(cleaned)
    except Exception:
        # If sanitizer fails, fall back to the cleaned dict.
        pass

    # 3) high-order suppression toward baseline
    base = getattr(reward_system, "base_coeffs", {}) or {}
    shrunk: Dict[str, float] = {}
    rel_sum = 0.0
    rel_sq_sum = 0.0
    rel_cnt = 0

    for k, v in cleaned.items():
        base_val = base.get(k)
        if base_val is None or base_val == 0.0:
            shrunk[k] = v
            continue

        m, n = _parse_mode_numbers(k)
        shrink = 1.0
        if enable_high_order_shrink:
            if (
                m is not None
                and n is not None
                and (m > int(high_order_m_threshold) or n > int(high_order_n_threshold))
            ):
                shrink = max(0.0, min(1.0, float(high_order_shrink)))

        v2 = float(base_val) + (float(v) - float(base_val)) * shrink
        shrunk[k] = v2
        rel = abs(v2 - float(base_val)) / (abs(float(base_val)) + 1e-12)
        rel_sum += rel
        rel_sq_sum += rel * rel
        rel_cnt += 1

    # 4) global step scaling (avoid "too wild" candidates)
    metric = str(global_step_metric or "sum").strip().lower()
    if metric == "avg":
        rel_total = rel_sum / max(1, int(rel_cnt))
    elif metric == "l2":
        rel_total = math.sqrt(rel_sq_sum)
    else:
        rel_total = rel_sum

    if max_total_rel_change is not None and max_total_rel_change > 0 and rel_total > max_total_rel_change:
        strategy = str(global_step_strategy or "scale").strip().lower()
        if strategy == "sparsify":
            keys = list(shrunk.keys())
            rng.shuffle(keys)
            kept: Dict[str, float] = {}
            rel_used = 0.0
            for k in keys:
                base_val = base.get(k)
                v = shrunk.get(k)
                if base_val is None or base_val == 0.0 or v is None:
                    kept[k] = v
                    continue
                rel_k = abs(float(v) - float(base_val)) / (abs(float(base_val)) + 1e-12)
                if rel_used + rel_k <= max_total_rel_change or not kept:
                    kept[k] = v
                    rel_used += rel_k
                else:
                    kept[k] = float(base_val)
            shrunk = kept
        else:
            scale = max_total_rel_change / (rel_total + 1e-12)
            scaled: Dict[str, float] = {}
            for k, v in shrunk.items():
                base_val = base.get(k)
                if base_val is None:
                    scaled[k] = v
                    continue
                scaled[k] = float(base_val) + (float(v) - float(base_val)) * float(scale)
            shrunk = scaled

    out = {"new_coefficients": shrunk}
    return canonicalize_json_obj(out)


def repair_gsco_decision(
    decision_json: str,
    reward_system,
    rng: random.Random,
    min_cells: int,
    max_cells: int,
    min_manhattan_dist: int,
    drop_isolated: bool,
    smooth_polarity: bool,
) -> Optional[str]:
    obj = parse_json_dict(decision_json)
    if obj is None:
        return None

    cells = obj.get("cells")
    if not isinstance(cells, list):
        return None

    nPhi = int(getattr(reward_system, "wf_nPhi", 12))
    nTheta = int(getattr(reward_system, "wf_nTheta", 12))
    forbidden = getattr(reward_system, "forbidden_cells", set()) or set()

    min_cells = int(min_cells) if min_cells is not None else 0

    cell_map: Dict[tuple, int] = {}
    for c in cells:
        if not isinstance(c, (list, tuple)) or len(c) != 3:
            continue
        try:
            phi = int(c[0]) % nPhi
            theta = int(c[1]) % nTheta
            state = int(c[2])
        except Exception:
            continue

        if (phi, theta) in forbidden:
            continue
        if state == 0:
            continue
        if state not in (-1, 1):
            state = 1 if state > 0 else -1
        cell_map[(phi, theta)] = state

    if not cell_map:
        if min_cells <= 0:
            return None
        # Fill with random valid cells (avoid forbidden) so the evaluator won't drop empty.
        while len(cell_map) < min_cells:
            phi = rng.randint(0, nPhi - 1)
            theta = rng.randint(0, nTheta - 1)
            if (phi, theta) in forbidden or (phi, theta) in cell_map:
                continue
            cell_map[(phi, theta)] = rng.choice([-1, 1])

    # Optional: enforce spacing
    if min_manhattan_dist is not None and int(min_manhattan_dist) > 0:
        dmin = int(min_manhattan_dist)
        keys = list(cell_map.keys())
        rng.shuffle(keys)
        kept: Dict[tuple, int] = {}
        for (phi, theta) in keys:
            ok = True
            for (p2, t2) in kept.keys():
                if abs(phi - p2) + abs(theta - t2) < dmin:
                    ok = False
                    break
            if ok:
                kept[(phi, theta)] = cell_map[(phi, theta)]
        cell_map = kept

    # Optional: drop isolated spikes
    if drop_isolated and len(cell_map) > max(1, min_cells):
        nbr4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        keys = list(cell_map.keys())
        isolated = []
        for (phi, theta) in keys:
            cnt = 0
            for dp, dt in nbr4:
                if ((phi + dp) % nPhi, (theta + dt) % nTheta) in cell_map:
                    cnt += 1
            if cnt == 0:
                isolated.append((phi, theta))
        rng.shuffle(isolated)
        for k in isolated:
            if len(cell_map) <= min_cells:
                break
            cell_map.pop(k, None)

    # Optional: smooth polarity using neighbor majority
    if smooth_polarity and len(cell_map) >= 3:
        nbr4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        new_map = dict(cell_map)
        for (phi, theta), st in cell_map.items():
            s = 0
            c = 0
            for dp, dt in nbr4:
                nb = ((phi + dp) % nPhi, (theta + dt) % nTheta)
                if nb in cell_map:
                    s += int(cell_map[nb])
                    c += 1
            if c >= 2 and abs(s) >= 2:
                new_map[(phi, theta)] = 1 if s > 0 else -1
        cell_map = new_map

    # Enforce max_cells
    if max_cells is not None and max_cells > 0 and len(cell_map) > int(max_cells):
        keys = list(cell_map.keys())
        rng.shuffle(keys)
        keys = keys[: int(max_cells)]
        cell_map = {k: cell_map[k] for k in keys}

    # Enforce min_cells by random fill (avoid evaluator dropping empty)
    while len(cell_map) < min_cells:
        phi = rng.randint(0, nPhi - 1)
        theta = rng.randint(0, nTheta - 1)
        if (phi, theta) in forbidden or (phi, theta) in cell_map:
            continue
        cell_map[(phi, theta)] = rng.choice([-1, 1])

    out_cells = [[p, t, cell_map[(p, t)]] for (p, t) in sorted(cell_map.keys())]
    out = {"cells": out_cells}
    return canonicalize_json_obj(out)
