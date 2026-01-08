import json
import math
import random
from typing import List, Tuple


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _parse_centers(decision_json: str, n_circles: int):
    try:
        obj = json.loads(decision_json)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    centers = obj.get("centers")
    if not isinstance(centers, list) or len(centers) != int(n_circles):
        return None
    out = []
    for c in centers:
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            return None
        try:
            x = float(c[0])
            y = float(c[1])
        except Exception:
            return None
        out.append((_clamp01(x), _clamp01(y)))
    return out


def _compute_radius_square(centers: List[Tuple[float, float]]) -> float:
    if not centers:
        return 0.0

    min_to_boundary = float("inf")
    for x, y in centers:
        d = min(x, 1.0 - x, y, 1.0 - y)
        if d < min_to_boundary:
            min_to_boundary = d

    min_pair = float("inf")
    n = len(centers)
    for i in range(n):
        xi, yi = centers[i]
        for j in range(i + 1, n):
            xj, yj = centers[j]
            dij = math.hypot(xi - xj, yi - yj)
            if dij < min_pair:
                min_pair = dij

    if min_pair == float("inf"):
        min_nonoverlap = float("inf")
    else:
        min_nonoverlap = 0.5 * min_pair

    r = min(min_to_boundary, min_nonoverlap)
    if r < 0.0:
        return 0.0
    return float(r)


def generate_initial_population(config, seed):
    random.seed(int(seed))

    pop_size = int(config.get("optimization.pop_size", 20) or 20)
    n_circles = int(config.get("circle_packing.n_circles", 10) or 10)

    out = []
    seen = set()
    attempts = 0
    max_attempts = pop_size * 50
    while len(out) < pop_size and attempts < max_attempts:
        attempts += 1
        centers = [[random.random(), random.random()] for _ in range(n_circles)]
        s = json.dumps({"centers": centers}, sort_keys=True, separators=(",", ":"))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.objs = config.get("goals", []) or []
        opt_dirs = config.get("optimization_direction", []) or []
        self.obj_directions = {
            obj: opt_dirs[i] if i < len(opt_dirs) else "min" for i, obj in enumerate(self.objs)
        }
        self.obj_ranges = config.get("objective_ranges", {}) or {}
        self.n_circles = int(config.get("circle_packing.n_circles", 10) or 10)

    def evaluate(self, items):
        invalid_num = 0

        r_min, r_max = 0.0, 0.5
        rr = self.obj_ranges.get("radius")
        if isinstance(rr, (list, tuple)) and len(rr) == 2:
            try:
                r_min = float(rr[0])
                r_max = float(rr[1])
            except Exception:
                r_min, r_max = 0.0, 0.5
        if not (r_max > r_min):
            r_min, r_max = 0.0, 0.5

        for it in items:
            centers = _parse_centers(getattr(it, "value", "") or "", self.n_circles)
            if centers is None:
                invalid_num += 1
                it.assign_results(
                    {
                        "original_results": {"radius": 0.0},
                        "transformed_results": {"radius": 1.0},
                        "overall_score": 0.0,
                        "constraint_results": {
                            "status": "invalid_input",
                            "sim_message": "invalid_centers_json",
                            "cv": 1e6,
                            "g1": 1e6,
                            "feasible": 0,
                        },
                    }
                )
                continue

            r = _compute_radius_square(centers)
            r_clip = r
            if r_clip < r_min:
                r_clip = r_min
            if r_clip > r_max:
                r_clip = r_max

            norm = (r_clip - r_min) / (r_max - r_min)
            score = norm
            if self.obj_directions.get("radius") == "max":
                score = 1.0 - norm

            it.assign_results(
                {
                    "original_results": {"radius": float(r)},
                    "transformed_results": {"radius": float(score)},
                    "overall_score": float(1.0 - score),
                    "constraint_results": {
                        "status": "ok",
                        "sim_message": "",
                        "cv": 0.0,
                        "g1": 0.0,
                        "feasible": 1,
                        "is_feasible": 1.0,
                    },
                }
            )

        return items, {"invalid_num": int(invalid_num), "repeated_num": 0}
