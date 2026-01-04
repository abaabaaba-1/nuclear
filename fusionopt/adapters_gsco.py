from __future__ import annotations

import json
import random
from typing import Optional

from .heu_repair import repair_gsco_decision
from .json_utils import canonicalize_json_obj, parse_json_dict


class GscoAdapter:
    def __init__(self, reward_system, config, rng: random.Random):
        self.reward_system = reward_system
        self.config = config
        self.rng = rng

    def gate(self, decision_json: str) -> Optional[str]:
        obj = parse_json_dict(decision_json)
        if obj is None:
            return None
        cells = obj.get("cells")
        if not isinstance(cells, list):
            return None
        return canonicalize_json_obj({"cells": cells})

    def heu_repair(self, decision_json: str) -> Optional[str]:
        min_cells = int(self.config.get("llm_constraints.min_active_cells", 3) or 3)
        max_cells = int(self.config.get("llm_constraints.max_active_cells", 60) or 60)
        min_dist = int(self.config.get("fusionopt.heu_repair.gsco.min_manhattan_dist", 0) or 0)
        drop_isolated = bool(self.config.get("fusionopt.heu_repair.gsco.drop_isolated", True))
        smooth_polarity = bool(self.config.get("fusionopt.heu_repair.gsco.smooth_polarity", True))
        return repair_gsco_decision(
            decision_json,
            reward_system=self.reward_system,
            rng=self.rng,
            min_cells=min_cells,
            max_cells=max_cells,
            min_manhattan_dist=min_dist,
            drop_isolated=drop_isolated,
            smooth_polarity=smooth_polarity,
        )

    def std_resample(self) -> str:
        nPhi = int(getattr(self.reward_system, "wf_nPhi", 12))
        nTheta = int(getattr(self.reward_system, "wf_nTheta", 12))
        min_cells = int(self.config.get("llm_constraints.min_active_cells", 3) or 3)
        max_cells = int(self.config.get("llm_constraints.max_active_cells", 60) or 60)
        n_active = self.rng.randint(min_cells, min(max_cells, 20))
        cell_map = {}
        while len(cell_map) < n_active:
            phi = self.rng.randint(0, nPhi - 1)
            theta = self.rng.randint(0, nTheta - 1)
            st = self.rng.choice([-1, 1])
            cell_map[(phi, theta)] = st
        cells = [[p, t, s] for (p, t), s in cell_map.items()]
        return json.dumps({"cells": cells})

    def std_mutation(self, parent_json: str) -> str:
        obj = parse_json_dict(parent_json) or {}
        cells = obj.get("cells") if isinstance(obj, dict) else None
        if not isinstance(cells, list):
            return self.std_resample()

        nPhi = int(getattr(self.reward_system, "wf_nPhi", 12))
        nTheta = int(getattr(self.reward_system, "wf_nTheta", 12))
        min_cells = int(self.config.get("llm_constraints.min_active_cells", 3) or 3)
        max_cells = int(self.config.get("llm_constraints.max_active_cells", 60) or 60)

        # canonicalize to map
        cell_map = {}
        for c in cells:
            if isinstance(c, (list, tuple)) and len(c) == 3:
                try:
                    phi, theta, state = int(c[0]), int(c[1]), int(c[2])
                except Exception:
                    continue
                phi %= nPhi
                theta %= nTheta
                if state == 0:
                    continue
                if state not in (-1, 1):
                    state = 1 if state > 0 else -1
                cell_map[(phi, theta)] = state

        mut_type = self.rng.choice([1, 2, 3, 4])
        keys = list(cell_map.keys())

        if mut_type == 1 and keys:  # flip
            k = self.rng.choice(keys)
            cell_map[k] *= -1
        elif mut_type == 2 and keys:  # move
            k = self.rng.choice(keys)
            st = cell_map.pop(k)
            phi, theta = k
            phi = (phi + self.rng.choice([-1, 0, 1])) % nPhi
            theta = (theta + self.rng.choice([-1, 0, 1])) % nTheta
            cell_map[(phi, theta)] = st
        elif mut_type == 3:  # add
            if len(cell_map) < max_cells:
                phi = self.rng.randint(0, nPhi - 1)
                theta = self.rng.randint(0, nTheta - 1)
                st = self.rng.choice([-1, 1])
                cell_map[(phi, theta)] = st
        elif mut_type == 4 and keys:  # remove
            if len(cell_map) > min_cells:
                k = self.rng.choice(keys)
                cell_map.pop(k, None)

        out_cells = [[p, t, s] for (p, t), s in cell_map.items()]
        return json.dumps({"cells": out_cells})

    def std_crossover(self, parent_a_json: str, parent_b_json: str) -> str:
        a = parse_json_dict(parent_a_json) or {}
        b = parse_json_dict(parent_b_json) or {}
        ca = a.get("cells") if isinstance(a, dict) else None
        cb = b.get("cells") if isinstance(b, dict) else None
        if not isinstance(ca, list) or not isinstance(cb, list):
            return self.std_resample()

        # sort for reproducible crossover
        ca2 = sorted([c for c in ca if isinstance(c, (list, tuple)) and len(c) == 3], key=lambda x: (x[0], x[1]))
        cb2 = sorted([c for c in cb if isinstance(c, (list, tuple)) and len(c) == 3], key=lambda x: (x[0], x[1]))
        min_len = min(len(ca2), len(cb2))
        if min_len < 2:
            child = ca2 if self.rng.random() < 0.5 else cb2
        else:
            cut = self.rng.randint(1, min_len - 1)
            child = ca2[:cut] + cb2[cut:]

        return json.dumps({"cells": child})
