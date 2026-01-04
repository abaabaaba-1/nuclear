import json
import random
import re


_COEFF_KEY_PATTERN = re.compile(r"([RZ]B[CS])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)


def _parse_mode_numbers(key: str):
    try:
        norm_key = key.strip().replace(" ", "")
        match = _COEFF_KEY_PATTERN.match(norm_key)
        if not match:
            return None, None
        return abs(int(match.group(2))), abs(int(match.group(3)))
    except Exception:
        return None, None


def propose(
    rng: random.Random,
    n: int,
    base_coeffs: dict,
    incumbent_json: str = "",
    max_coeff_changes: int = 12,
    low_order_max_rel_change: float = 0.02,
    high_order_max_rel_change: float = 0.05,
):
    """Return a list of candidate JSON strings.

    The VMEC RewardingSystem interprets `new_coefficients` as *absolute* values
    for the selected keys; missing keys mean "use baseline".
    """

    try:
        keys = list(base_coeffs.keys())
    except Exception:
        keys = []

    incumbent_delta = {}
    if incumbent_json:
        try:
            payload = json.loads(incumbent_json)
            incumbent_delta = payload.get("new_coefficients") or {}
            if not isinstance(incumbent_delta, dict):
                incumbent_delta = {}
        except Exception:
            incumbent_delta = {}

    out = []
    for _ in range(max(int(n), 1)):
        # Start from incumbent (sometimes) so evolution can refine.
        current = dict(incumbent_delta) if (incumbent_delta and rng.random() < 0.6) else {}

        if not keys:
            out.append(json.dumps({"new_coefficients": current}))
            continue

        num_to_mutate = rng.randint(1, max(1, int(max_coeff_changes)))
        chosen = rng.sample(keys, min(num_to_mutate, len(keys)))

        for k in chosen:
            base_val = base_coeffs.get(k)
            if base_val is None:
                continue
            try:
                base_val = float(base_val)
            except Exception:
                continue
            if base_val == 0.0:
                continue

            m, nn = _parse_mode_numbers(k)
            rel_limit = float(high_order_max_rel_change)
            if m is not None and nn is not None and m <= 2 and nn <= 1:
                rel_limit = float(low_order_max_rel_change)

            rel = rng.uniform(-rel_limit, rel_limit)
            current[k] = base_val * (1.0 + float(rel))

        out.append(json.dumps({"new_coefficients": current}))

    return out
