import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import List

import numpy as np
import yaml
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from algorithm.base import ItemFactory
from model.util import top_auc

from eval_logger import EvalLogger


class Config:
    """Lightweight config wrapper supporting dotted keys (e.g. 'coil_design.wf_nPhi')."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default=None):
        if not isinstance(key, str):
            return default
        keys = key.split(".")
        val = self._data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def to_string(self) -> str:
        return yaml.dump(self._data)


class GSCOProblem(Problem):
    """pymoo Problem wrapper for GSCO-Lite coil design.

    Decision variable: a single JSON string with the format {"cells": [[phi, theta, state], ...]}.
    Objectives: transformed multi-objective scores returned by the GSCO-Lite RewardingSystem,
    already normalized and set up for minimization.
    """

    def __init__(self, reward_system, config: Config, item_factory: ItemFactory, eval_logger=None, pop_size: int | None = None):
        self.reward_system = reward_system
        self.config = config
        self.item_factory = item_factory
        self.eval_logger = eval_logger
        self.pop_size = int(pop_size) if pop_size is not None else None
        self.history_buffer: List = []  # (Item, eval_index)
        self.eval_count: int = 0

        n_obj = len(config.get("goals"))
        super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=0, elementwise=False, vtype=object)

    def _evaluate(self, X, out, *args, **kwargs):
        t0 = time.time()
        json_list = [row[0] for row in X]
        items = [self.item_factory.create(s) for s in json_list]

        n_obj = len(self.config.get("goals"))
        penalty_scores = [1.0e6] * int(n_obj)

        # IMPORTANT: pymoo expects out["F"] to have shape (len(X), n_obj).
        # The RewardingSystem may drop or fail candidates; to keep shapes consistent,
        # evaluate candidates one-by-one and assign penalty scores on failure.
        evaluated_items = []
        for it in items:
            try:
                tmp_items, _ = self.reward_system.evaluate([it])
                if not tmp_items:
                    raise RuntimeError("Empty evaluation result")
                ev = tmp_items[0]
            except Exception:
                ev = it

            try:
                if getattr(ev, "scores", None) is None or len(ev.scores) != n_obj:
                    ev.scores = list(penalty_scores)
                if getattr(ev, "total", None) is None:
                    ev.total = -1.0
                if getattr(ev, "property", None) is None:
                    ev.property = {}
                if getattr(ev, "constraints", None) is None:
                    ev.constraints = 0.0
            except Exception:
                pass

            evaluated_items.append(ev)

        dt = time.time() - t0

        start_eval = self.eval_count
        self.eval_count += len(evaluated_items)
        for i, it in enumerate(evaluated_items):
            self.history_buffer.append((it, start_eval + i + 1))

        generation = -1
        try:
            if self.pop_size and self.pop_size > 0:
                generation = int(start_eval // self.pop_size)
        except Exception:
            generation = -1

        if self.eval_logger is not None:
            self.eval_logger.log_batch(evaluated_items, generation=generation, total_time_sec=dt, tier="true")

        out["F"] = np.array([it.scores for it in evaluated_items], dtype=float)


class CellsCrossover(Crossover):
    """One-point crossover on cell lists in the GSCO-Lite JSON design.

    Parents and offsprings are JSON strings with a "cells" list.
    We sort cells by (phi, theta), perform one-point crossover, and then
    deduplicate cells so that at most one state exists per grid cell.
    """

    def __init__(self, nPhi: int, nTheta: int):
        super().__init__(2, 2)
        self.nPhi = nPhi
        self.nTheta = nTheta

    def _do(self, problem, X, **kwargs):  # X: (n_parents, n_matings, n_var)
        n_matings = X.shape[1]
        offspring = np.empty((self.n_offsprings, n_matings, problem.n_var), dtype=object)

        for m in range(n_matings):
            parent_a = json.loads(X[0, m, 0])
            parent_b = json.loads(X[1, m, 0])

            cells_a = parent_a.get("cells", []) or []
            cells_b = parent_b.get("cells", []) or []

            # Sort by (phi, theta) for reproducible crossover
            cells_a = sorted(cells_a, key=lambda c: (c[0], c[1]))
            cells_b = sorted(cells_b, key=lambda c: (c[0], c[1]))

            min_len = min(len(cells_a), len(cells_b))
            if min_len < 2:
                child1_cells = cells_a
                child2_cells = cells_b
            else:
                cut = random.randint(1, min_len - 1)
                child1_cells = cells_a[:cut] + cells_b[cut:]
                child2_cells = cells_b[:cut] + cells_a[cut:]

            def dedup(cells):
                cell_map = {}
                for c in cells:
                    if not isinstance(c, (list, tuple)) or len(c) != 3:
                        continue
                    phi, theta, state = int(c[0]), int(c[1]), int(c[2])
                    phi = phi % self.nPhi
                    theta = theta % self.nTheta
                    if state == 0:
                        continue
                    if state not in (-1, 1):
                        state = 1 if state > 0 else -1
                    cell_map[(phi, theta)] = state
                return [[p, t, s] for (p, t), s in cell_map.items()]

            child1_cells = dedup(child1_cells)
            child2_cells = dedup(child2_cells)

            offspring[0, m, 0] = json.dumps({"cells": child1_cells})
            offspring[1, m, 0] = json.dumps({"cells": child2_cells})

        return offspring


class CellsMutation(Mutation):
    """Mutation operator on GSCO-Lite cell designs.

    Matches the spirit of the StandardGA/SA mutations in run_gsco_baselines.py:
    - type 1: flip polarity
    - type 2: move a cell by a small random walk in (phi, theta)
    - type 3: add a new active cell (up to max_cells)
    - type 4: remove an active cell (down to min_cells)
    """

    def __init__(self, nPhi: int, nTheta: int, min_cells: int, max_cells: int):
        super().__init__()
        self.nPhi = nPhi
        self.nTheta = nTheta
        self.min_cells = min_cells
        self.max_cells = max_cells

    def _do(self, problem, X, **kwargs):  # X: (n_individuals, n_var)
        mutated = np.empty_like(X, dtype=object)

        for idx in range(len(X)):
            design = json.loads(X[idx, 0])
            cells = design.get("cells", []) or []

            mut_type = random.choice([1, 2, 3, 4])

            if mut_type == 1 and cells:  # Flip
                i = random.randrange(len(cells))
                cells[i][2] *= -1

            elif mut_type == 2 and cells:  # Move
                i = random.randrange(len(cells))
                cells[i][0] = (cells[i][0] + random.choice([-1, 0, 1])) % self.nPhi
                cells[i][1] = (cells[i][1] + random.choice([-1, 0, 1])) % self.nTheta

            elif mut_type == 3:  # Add
                if len(cells) < self.max_cells:
                    phi = random.randint(0, self.nPhi - 1)
                    theta = random.randint(0, self.nTheta - 1)
                    state = random.choice([-1, 1])
                    cells.append([phi, theta, state])

            elif mut_type == 4 and cells:  # Remove
                if len(cells) > self.min_cells:
                    i = random.randrange(len(cells))
                    cells.pop(i)

            # Deduplicate and enforce state in {-1, 1}
            cell_map = {}
            for c in cells:
                if not isinstance(c, (list, tuple)) or len(c) != 3:
                    continue
                phi, theta, state = int(c[0]), int(c[1]), int(c[2])
                phi = phi % self.nPhi
                theta = theta % self.nTheta
                if state == 0:
                    continue
                if state not in (-1, 1):
                    state = 1 if state > 0 else -1
                cell_map[(phi, theta)] = state

            final_cells = [[p, t, s] for (p, t), s in cell_map.items()]

            # Ensure at least min_cells by randomly activating new cells if needed
            while len(final_cells) < self.min_cells:
                phi = random.randint(0, self.nPhi - 1)
                theta = random.randint(0, self.nTheta - 1)
                if (phi, theta) in cell_map:
                    continue
                state = random.choice([-1, 1])
                cell_map[(phi, theta)] = state
                final_cells = [[p, t, s] for (p, t), s in cell_map.items()]

            mutated[idx, 0] = json.dumps({"cells": final_cells})

        return mutated


def run_gsco_pymoo(config_path: str, algo: str, seed: int) -> None:
    """Run a Pymoo-based multi-objective baseline on GSCO-Lite.

    Supported algorithms (algo): 'nsga2', 'sms', 'moead', 'rvea'.
    """

    if not (config_path.startswith("problem/") or os.path.isabs(config_path)):
        config_path = os.path.join("problem", config_path)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = Config(config_data)

    random.seed(seed)
    np.random.seed(seed)

    property_list = config.get("goals")
    if 'logging' not in config_data or not isinstance(config_data.get('logging'), dict):
        config_data['logging'] = {}
    problem_id = config.get('protocol.problem_id', 'stellarator_coil_gsco_lite')
    config_data['logging']['results_base_dir'] = os.path.join('./results', str(problem_id))

    module_path = config.get("evalutor_path")
    module = __import__(module_path, fromlist=["RewardingSystem", "generate_initial_population"])
    RewardingSystem = getattr(module, "RewardingSystem")
    generate_initial_population = getattr(module, "generate_initial_population")

    reward_system = RewardingSystem(config)
    item_factory = ItemFactory(property_list)

    # 1. Initial population
    print("Generating initial population for GSCO-Lite Pymoo baseline...")
    initial_population_strs = generate_initial_population(config, seed)
    if not initial_population_strs:
        raise RuntimeError("Initial population generator returned no candidates.")

    pop_size = config.get("optimization.pop_size")
    eval_budget = config.get("optimization.eval_budget")
    log_freq = config.get("optimization.log_freq", 50)

    eval_logger = None
    try:
        algo_id = config.get('protocol.algo_id', None) or str(algo)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        run_name = config.get('exper_name', 'run')
        run_id = f"{run_name}_seed{seed}_{ts}"
        run_dir = os.path.join(config_data['logging']['results_base_dir'], algo_id, run_id)
        eval_logger = EvalLogger(run_dir, problem_id, algo_id, seed, run_id, config_data, property_list)
    except Exception:
        eval_logger = None

    problem = GSCOProblem(reward_system, config, item_factory, eval_logger=eval_logger, pop_size=pop_size)

    nPhi = config.get("coil_design.wf_nPhi", 12)
    nTheta = config.get("coil_design.wf_nTheta", 12)
    min_cells = config.get("llm_constraints.min_active_cells", 3)
    max_cells = config.get("llm_constraints.max_active_cells", 60)

    initial_sampling = np.array(initial_population_strs, dtype=object).reshape(-1, 1)
    crossover = CellsCrossover(nPhi, nTheta)
    mutation = CellsMutation(nPhi, nTheta, min_cells, max_cells)

    n_obj = len(property_list)

    # 2. Select algorithm and termination
    algo = algo.lower()
    if algo == "nsga2":
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=initial_sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )
        termination = ("n_eval", eval_budget)

    elif algo == "sms":
        algorithm = SMSEMOA(
            pop_size=pop_size,
            sampling=initial_sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )
        termination = ("n_eval", eval_budget)

    elif algo == "moead":
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)
        algorithm = MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=initial_sampling,
            crossover=crossover,
            mutation=mutation,
        )
        termination = ("n_eval", eval_budget)

    elif algo == "rvea":
        n_partitions = config.get("baseline_rvea.n_partitions", 12)
        ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=n_partitions)
        algorithm = RVEA(
            ref_dirs=ref_dirs,
            sampling=initial_sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )

        # Approximate number of generations from eval_budget and pop_size
        if pop_size is None or pop_size <= 0:
            approx_n_gen = max(eval_budget, 1)
        else:
            approx_n_gen = max(eval_budget // pop_size, 1)
        termination = ("n_gen", int(approx_n_gen))

    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # 3. Run optimization
    print(f"Running GSCO-Lite {algo.upper()} baseline (budget={eval_budget})...")
    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
    )

    running_time = time.time() - start_time
    print(f"Optimization finished in {running_time / 3600:.2f} hours.")

    # 4. Process and save results
    all_evaluated = problem.history_buffer  # list of (Item, eval_idx)

    evaluated_initial_items = [item for item, _ in all_evaluated[: len(initial_population_strs)]]

    # Extract final population items corresponding to the Pymoo solution set
    try:
        solution_values = {individual[0] for individual in res.X}
    except Exception:
        solution_values = set()

    if solution_values:
        final_population_items = [item for item, _ in all_evaluated if item.value in solution_values]
    else:
        final_population_items = [item for item, _ in all_evaluated]

    results_log = {"results": [], "params": config.to_string()}

    for i in range(log_freq, eval_budget + 1, log_freq):
        subset_buffer = [pair for pair in all_evaluated if pair[1] <= i]
        if not subset_buffer:
            continue
        top100_items = sorted(
            [it for it, _ in subset_buffer], key=lambda x: x.total, reverse=True
        )[:100]
        results_log["results"].append(
            {
                "Training_step": i,
                "all_unique_moles": len(subset_buffer),
                "avg_top1": top100_items[0].total if top100_items else -1,
                "avg_top10": float(
                    np.mean([item.total for item in top100_items[:10]])
                )
                if top100_items
                else -1,
                "avg_top100": float(
                    np.mean([item.total for item in top100_items])
                )
                if top100_items
                else -1,
                "top1_auc": top_auc(subset_buffer, 1, False, 100, eval_budget),
                "top10_auc": top_auc(subset_buffer, 10, False, 100, eval_budget),
                "top100_auc": top_auc(subset_buffer, 100, False, 100, eval_budget),
                "running_time[s]": running_time * (i / eval_budget),
            }
        )

    final_top100 = sorted(
        [it for it, _ in all_evaluated], key=lambda x: x.total, reverse=True
    )[:100]
    results_log["results"].append(
        {
            "Training_step": len(all_evaluated),
            "all_unique_moles": len(all_evaluated),
            "avg_top1": final_top100[0].total if final_top100 else -1,
            "avg_top10": float(
                np.mean([item.total for item in final_top100[:10]])
            )
            if final_top100
            else -1,
            "avg_top100": float(
                np.mean([item.total for item in final_top100])
            )
            if final_top100
            else -1,
            "top1_auc": top_auc(all_evaluated, 1, True, 100, eval_budget),
            "top10_auc": top_auc(all_evaluated, 10, True, 100, eval_budget),
            "top100_auc": top_auc(all_evaluated, 100, True, 100, eval_budget),
            "running_time[s]": running_time,
        }
    )

    try:
        if eval_logger is not None:
            eval_logger.close()
    except Exception:
        pass

    try:
        if eval_logger is not None:
            from check_phase0_protocol import _repair_run
            from pathlib import Path

            _repair_run(Path(str(run_dir)))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Pymoo-based multi-objective baselines for GSCO-Lite (coil design)"
    )
    parser.add_argument(
        "algo",
        type=str,
        choices=["nsga2", "sms", "moead", "rvea"],
        help="Which algorithm to run (NSGA-II, SMS-EMOA, MOEA/D, RVEA)",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="problem/stellarator_coil_gsco_lite/config.yaml",
        help="Path to GSCO-Lite config YAML",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_gsco_pymoo(args.config, args.algo, args.seed)


if __name__ == "__main__":
    main()
