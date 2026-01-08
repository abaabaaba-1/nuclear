from __future__ import annotations

import json
import random
from typing import Optional

from .json_utils import canonicalize_json_obj, parse_json_dict


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class CirclePackingAdapter:
    def __init__(self, reward_system, config, rng: random.Random):
        self.reward_system = reward_system
        self.config = config
        self.rng = rng

    def _n_circles(self) -> int:
        return int(self.config.get("circle_packing.n_circles", 10) or 10)

    def gate(self, decision_json: str) -> Optional[str]:
        obj = parse_json_dict(decision_json)
        if obj is None:
            return None
        centers = obj.get("centers")
        n = self._n_circles()
        if not isinstance(centers, list) or len(centers) != n:
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
            out.append([_clamp01(x), _clamp01(y)])

        return canonicalize_json_obj({"centers": out})

    def heu_repair(self, decision_json: str) -> Optional[str]:
        return self.gate(decision_json)

    def std_resample(self) -> str:
        n = self._n_circles()
        centers = [[self.rng.random(), self.rng.random()] for _ in range(n)]
        return json.dumps({"centers": centers})

    def std_mutation(self, parent_json: str) -> str:
        obj = parse_json_dict(parent_json) or {}
        centers = obj.get("centers") if isinstance(obj, dict) else None
        n = self._n_circles()
        if not isinstance(centers, list) or len(centers) != n:
            return self.std_resample()

        sigma = float(self.config.get("fusionopt.std_mutation.circle_packing.sigma", 0.05) or 0.05)
        kmax = int(self.config.get("fusionopt.std_mutation.circle_packing.max_points", 3) or 3)
        kmax = max(1, min(kmax, n))
        k = self.rng.randint(1, kmax)
        idxs = self.rng.sample(list(range(n)), k)

        out = []
        for i, c in enumerate(centers):
            try:
                x = float(c[0])
                y = float(c[1])
            except Exception:
                x, y = self.rng.random(), self.rng.random()
            if i in idxs:
                x += self.rng.gauss(0.0, sigma)
                y += self.rng.gauss(0.0, sigma)
            out.append([_clamp01(x), _clamp01(y)])

        return json.dumps({"centers": out})

    def std_crossover(self, parent_a_json: str, parent_b_json: str) -> str:
        a = parse_json_dict(parent_a_json) or {}
        b = parse_json_dict(parent_b_json) or {}
        ca = a.get("centers") if isinstance(a, dict) else None
        cb = b.get("centers") if isinstance(b, dict) else None
        n = self._n_circles()
        if not isinstance(ca, list) or not isinstance(cb, list) or len(ca) != n or len(cb) != n:
            return self.std_resample()

        out = []
        for i in range(n):
            src = ca if (self.rng.random() < 0.5) else cb
            c = src[i]
            try:
                x = float(c[0])
                y = float(c[1])
            except Exception:
                x, y = self.rng.random(), self.rng.random()
            out.append([_clamp01(x), _clamp01(y)])

        return json.dumps({"centers": out})
