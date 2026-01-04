import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import List

import numpy as np
import yaml
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from algorithm.base import ItemFactory

from baseline_gsco_pymoo import CellsCrossover, CellsMutation

try:
    from problem.stellarator_vmec.vmec_reset_helper import maybe_reset_vmec_inputs
except ImportError:
    def maybe_reset_vmec_inputs(*_args, **_kwargs):
        return

from eval_logger import EvalLogger


class BudgetExhausted(Exception):
    pass


class Config:
    """Lightweight config wrapper compatible with existing baseline scripts."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def to_string(self):
        return yaml.dump(self._data)


class MOLLMProblem(Problem):
    """Generic pymoo Problem wrapper for MOLLM-style evaluators.

    Each decision variable is a single JSON string describing design modifications
    (e.g., {"new_coefficients": {...}}). Evaluation is delegated to the existing
    RewardingSystem, and the **transformed multi-objective scores** are used as
    pymoo objectives (minimization).
    """

    def __init__(self, reward_system, config: Config, item_factory: ItemFactory, eval_logger=None, pop_size=None, eval_budget=None):
        self.reward_system = reward_system
        self.config = config
        self.item_factory = item_factory
        self.history_buffer: List = []  # (Item, eval_index)
        self.eval_count: int = 0
        self.eval_budget = int(eval_budget) if eval_budget is not None else None
        self.eval_logger = eval_logger
        self.pop_size = pop_size

        n_obj = len(config.get("goals"))
        super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=0, elementwise=True, vtype=object)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.eval_budget is not None and self.eval_count >= self.eval_budget:
            raise BudgetExhausted(f"Evaluation budget exhausted ({self.eval_count}/{self.eval_budget})")
        # x is an array-like of length 1 containing the JSON string
        candidate_json_str = x[0]
        item = self.item_factory.create(candidate_json_str)
        t0 = time.time()
        evaluated_items, _ = self.reward_system.evaluate([item])
        dt = time.time() - t0
        evaluated_item = evaluated_items[0]

        self.eval_count += 1
        self.history_buffer.append((evaluated_item, self.eval_count))

        # evaluator writes normalized, minimization-ready objectives into item.scores
        out["F"] = np.array(evaluated_item.scores, dtype=float)

        if getattr(self, 'eval_logger', None) is not None:
            gen_val = -1
            try:
                if self.pop_size is not None and int(self.pop_size) > 0:
                    gen_val = int((self.eval_count - 1) // int(self.pop_size))
            except Exception:
                gen_val = -1
            self.eval_logger.log_batch([evaluated_item], generation=gen_val, total_time_sec=dt, tier="true")


class JSONUniformCrossover(Crossover):
    """Uniform crossover on JSON designs with "new_coefficients".

    This mirrors the crossover used in the MOEA/D baseline, operating on the
    union of coefficient keys present in the two parents. Missing keys are
    allowed and handled gracefully.
    """

    def __init__(self, coeff_keys: List[str]):
        super().__init__(2, 2)
        self.coeff_keys = coeff_keys

    def _do(self, problem, X, **kwargs):  # X: (n_parents, n_matings, n_var)
        n_matings = X.shape[1]
        offspring = np.empty((self.n_offsprings, n_matings, problem.n_var), dtype=object)

        for m in range(n_matings):
            parent_a = json.loads(X[0, m, 0])
            parent_b = json.loads(X[1, m, 0])

            child1 = {"new_coefficients": {}}
            child2 = {"new_coefficients": {}}

            pa = parent_a.get("new_coefficients", {}) or {}
            pb = parent_b.get("new_coefficients", {}) or {}
            keys = set(pa.keys()) | set(pb.keys())
            if not keys:
                keys = set(self.coeff_keys)

            for key in keys:
                val_a = pa.get(key)
                val_b = pb.get(key)
                if val_a is None and val_b is None:
                    continue
                if val_a is None:
                    val_a = val_b
                if val_b is None:
                    val_b = val_a

                if random.random() < 0.5:
                    child1["new_coefficients"][key] = val_a
                    child2["new_coefficients"][key] = val_b
                else:
                    child1["new_coefficients"][key] = val_b
                    child2["new_coefficients"][key] = val_a

            offspring[0, m, 0] = json.dumps(child1)
            offspring[1, m, 0] = json.dumps(child2)

        return offspring


class JSONScalingMutation(Mutation):
    """Coefficient-wise scaling mutation on JSON designs.

    For each design, randomly pick a subset of existing keys in
    "new_coefficients" and scale their values by a random factor in
    [low, high]. This mirrors the behavior in the MOEA/D baseline.
    """

    def __init__(self, coeff_keys: List[str], factor_range):
        super().__init__()
        self.coeff_keys = coeff_keys
        self.low, self.high = factor_range

    def _do(self, problem, X, **kwargs):  # X: (n_individuals, n_var)
        mutated = np.empty_like(X, dtype=object)

        for idx in range(len(X)):
            design = json.loads(X[idx, 0])
            coeffs = design.setdefault("new_coefficients", {})
            if not coeffs:
                mutated[idx, 0] = json.dumps(design)
                continue

            num_mutations = random.randint(1, max(1, len(coeffs)))
            mutate_keys = random.sample(list(coeffs.keys()), num_mutations)

            for key in mutate_keys:
                try:
                    coeffs[key] = float(coeffs[key]) * random.uniform(self.low, self.high)
                except (TypeError, ValueError):
                    # If value is not numeric, skip mutation on this key
                    continue

            mutated[idx, 0] = json.dumps(design)

        return mutated


def main():
    parser = argparse.ArgumentParser(description="Run baseline RVEA for MOLLM problems")
    parser.add_argument(
        "config",
        nargs="?",
        default="sacs_geo_jk/config.yaml",
        help="Path to config YAML (e.g., problem/stellarator_vmec/config_moead.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Load configuration
    config_path = args.config
    if not (config_path.startswith("problem/") or os.path.isabs(config_path)):
        config_path = os.path.join("problem", config_path)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    config = Config(config_data)
    seed = args.seed

    if 'protocol' not in config_data or not isinstance(config_data.get('protocol'), dict):
        config_data['protocol'] = {}
    config_data['protocol'].setdefault('algo_id', 'rvea')

    if 'logging' not in config_data or not isinstance(config_data.get('logging'), dict):
        config_data['logging'] = {}
    problem_id = config.get('protocol.problem_id', None)
    if problem_id:
        config_data['logging']['results_base_dir'] = os.path.join('./results', str(problem_id))

    # 2. Setup environment
    random.seed(seed)
    np.random.seed(seed)

    property_list = config.get("goals")

    module_path = config.get("evalutor_path")
    module = __import__(module_path, fromlist=["RewardingSystem", "generate_initial_population"])
    RewardingSystem = getattr(module, "RewardingSystem")
    generate_initial_population = getattr(module, "generate_initial_population")

    eval_logger = None
    evalutor_path = config.get('evalutor_path')
    project_path = config.get('vmec.project_path')
    maybe_reset_vmec_inputs(evalutor_path, project_path, 'pre')
    try:
        problem_id = config.get('protocol.problem_id', None)
        results_base_dir = os.path.join('./results', str(problem_id)) if problem_id else None
        algo_id = config.get('protocol.algo_id', None)
        if problem_id and results_base_dir and algo_id:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            run_name = config.get('exper_name', 'run')
            run_id = f"{run_name}_seed{seed}_{ts}"
            run_dir = os.path.join(results_base_dir, algo_id, run_id)
            eval_logger = EvalLogger(run_dir, problem_id, algo_id, seed, run_id, config_data, property_list)

        reward_system = RewardingSystem(config)
        item_factory = ItemFactory(property_list)

        # 3. Initial population
        print("Generating initial population for RVEA baseline...")
        initial_population_strs = generate_initial_population(config, seed)

        if not initial_population_strs:
            raise RuntimeError("Initial population generator returned no candidates.")

        eval_budget = config.get("optimization.eval_budget")
        if eval_budget is None:
            raise RuntimeError("Missing required config key: optimization.eval_budget")
        eval_budget = int(eval_budget)
        if eval_budget <= 0:
            raise RuntimeError("optimization.eval_budget must be a positive integer")

        log_freq = config.get("optimization.log_freq")
        pop_size = config.get("optimization.pop_size")

        if pop_size is None or int(pop_size) <= 0:
            pop_size = len(initial_population_strs)
        else:
            pop_size = int(pop_size)

        if pop_size > eval_budget:
            pop_size = eval_budget

        if len(initial_population_strs) < pop_size:
            extra_needed = pop_size - len(initial_population_strs)
            initial_population_strs = (initial_population_strs * ((extra_needed // len(initial_population_strs)) + 1))[:pop_size]
        elif len(initial_population_strs) > pop_size:
            initial_population_strs = initial_population_strs[:pop_size]

        problem = MOLLMProblem(reward_system, config, item_factory, eval_logger=eval_logger, pop_size=pop_size, eval_budget=eval_budget)

        initial_sampling = np.array(initial_population_strs, dtype=object).reshape(pop_size, 1)

        sample_payload = None
        try:
            sample_payload = json.loads(initial_population_strs[0])
        except Exception:
            sample_payload = None

        if isinstance(sample_payload, dict) and "cells" in sample_payload:
            nPhi = int(config.get("coil_design.wf_nPhi", 12))
            nTheta = int(config.get("coil_design.wf_nTheta", 12))
            min_cells = int(config.get("llm_constraints.min_active_cells", 3))
            max_cells = int(config.get("llm_constraints.max_active_cells", 60))
            crossover_op = CellsCrossover(nPhi, nTheta)
            mutation_op = CellsMutation(nPhi, nTheta, min_cells, max_cells)
        else:
            coeff_key_pool = sorted(
                {
                    key
                    for cand_str in initial_population_strs
                    for key in json.loads(cand_str).get("new_coefficients", {}).keys()
                }
            )

            mutation_factor_range = config.get("baseline.mutation_factor_range", [0.85, 1.15])
            crossover_op = JSONUniformCrossover(coeff_key_pool)
            mutation_op = JSONScalingMutation(coeff_key_pool, mutation_factor_range)

        # 4. Setup RVEA algorithm
        n_obj = len(property_list)
        n_partitions = config.get("baseline_rvea.n_partitions", 12)

        ref_dirs = None
        try:
            ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=int(n_partitions))
        except Exception:
            ref_dirs = None

        if ref_dirs is None or len(ref_dirs) != pop_size:
            try:
                ref_dirs = get_reference_directions("energy", n_obj, n_points=int(pop_size))
            except Exception:
                ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=int(n_partitions))

        algorithm = RVEA(
            ref_dirs=ref_dirs,
            pop_size=int(pop_size),
            sampling=initial_sampling,
            crossover=crossover_op,
            mutation=mutation_op,
            # IMPORTANT: decision variables are JSON strings (vtype=object),
            # so we must disable pymoo's default duplicate elimination based on
            # Euclidean distance in decision space (which assumes floats).
            eliminate_duplicates=False,
        )

        # In this pymoo version, RVEA is coupled with a generation-based
        # termination (MaxGen). Using an "n_eval" termination leads to an
        # invalid progress value inside MaxGen (n_max_gen == 0). Instead we
        # approximate the number of generations from the evaluation budget
        # and population size.

        # Each generation evaluates roughly `pop_size` individuals (after the
        # initial population). A simple, robust choice is:
        #   n_gen ≈ max(eval_budget // pop_size, 1)
        # which keeps total evaluations on the same order as eval_budget
        # without relying on pymoo's internal accounting.
        try:
            n_offsprings = int(getattr(algorithm, "n_offsprings", None) or 0)
        except Exception:
            n_offsprings = 0
        if n_offsprings <= 0:
            n_offsprings = int(pop_size) if pop_size else 1

        if pop_size is None or pop_size <= 0:
            approx_n_gen = max(eval_budget, 1)
        else:
            rem = max(int(eval_budget) - int(pop_size), 0)
            approx_n_gen = 1 + ((rem + n_offsprings - 1) // n_offsprings)
            approx_n_gen = max(int(approx_n_gen), 1)

        termination = ("n_gen", int(approx_n_gen))

        # 5. Run optimization
        print(f"Running RVEA baseline for {eval_budget} evaluations...")
        start_time = time.time()

        try:
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=seed,
                verbose=True,
                save_history=False,
            )
        except BudgetExhausted:
            res = None

        running_time = time.time() - start_time
        print(f"Optimization finished in {running_time / 3600:.2f} hours.")
        try:
            actual_evals = int(getattr(problem, "eval_count", 0))
            print(f"Total true evaluations: {actual_evals} (budget={eval_budget})")
        except Exception:
            pass
    finally:
        try:
            if eval_logger is not None:
                eval_logger.close()
        except Exception:
            pass
        maybe_reset_vmec_inputs(evalutor_path, project_path, 'post')


if __name__ == "__main__":
    main()
