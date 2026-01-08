import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from algorithm.base import ItemFactory
from eval_logger import EvalLogger
from fusionopt.protocol import ensure_protocol_fields

try:
    from problem.stellarator_vmec.vmec_reset_helper import maybe_reset_vmec_inputs
except Exception:
    def maybe_reset_vmec_inputs(*_args, **_kwargs):
        return


class Config:
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
    def __init__(self, reward_system, config: Config, item_factory: ItemFactory, eval_logger: EvalLogger, pop_size: int):
        self.reward_system = reward_system
        self.config = config
        self.item_factory = item_factory
        self.eval_logger = eval_logger
        self.pop_size = int(pop_size)
        self.eval_count = 0

        n_obj = len(config.get("goals"))
        super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=0, elementwise=True, vtype=object)

    def _evaluate(self, x, out, *args, **kwargs):
        candidate_json_str = x[0]
        item = self.item_factory.create(candidate_json_str)

        t0 = time.time()
        evaluated_items, _ = self.reward_system.evaluate([item])
        dt = time.time() - t0

        if evaluated_items:
            item = evaluated_items[0]

        gen_val = -1
        try:
            if self.pop_size > 0:
                gen_val = int(self.eval_count // self.pop_size)
        except Exception:
            gen_val = -1

        self.eval_count += 1

        try:
            ensure_protocol_fields(self.eval_logger.problem_id, [item])
        except Exception:
            pass

        try:
            self.eval_logger.log_batch([item], generation=gen_val, total_time_sec=dt, tier="true")
        except Exception:
            pass

        scores = getattr(item, "scores", None)
        if scores is None:
            scores = [1.0 for _ in range(self.n_obj)]
        out["F"] = np.array(scores, dtype=float)


class JSONUniformCrossover(Crossover):
    def __init__(self, coeff_keys):
        super().__init__(2, 2)
        self.coeff_keys = list(coeff_keys)

    def _do(self, problem, X, **kwargs):
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

            for key in sorted(keys):
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
    def __init__(self, coeff_keys, factor_range):
        super().__init__()
        self.coeff_keys = list(coeff_keys)
        self.low, self.high = float(factor_range[0]), float(factor_range[1])

    def _do(self, problem, X, **kwargs):
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
                    continue

            mutated[idx, 0] = json.dumps(design)

        return mutated


def main():
    parser = argparse.ArgumentParser(description="Run baseline NSGA-II for MOLLM problems (pymoo-based, protocol logging).")
    parser.add_argument(
        "config",
        nargs="?",
        default="sacs_geo_jk/config.yaml",
        help="Path to config YAML (e.g., problem/stellarator_vmec/config_baseline_nsga2_budget1000.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    config_path = args.config
    if not (config_path.startswith("problem/") or os.path.isabs(config_path)):
        config_path = os.path.join("problem", config_path)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = Config(config_data)
    seed = int(args.seed)

    random.seed(seed)
    np.random.seed(seed)

    if "protocol" not in config_data or not isinstance(config_data.get("protocol"), dict):
        config_data["protocol"] = {}
    config_data["protocol"].setdefault("problem_id", config.get("protocol.problem_id", None) or "stellarator_vmec")
    config_data["protocol"].setdefault("algo_id", config.get("protocol.algo_id", None) or "nsga2")

    if "logging" not in config_data or not isinstance(config_data.get("logging"), dict):
        config_data["logging"] = {}

    problem_id = config_data["protocol"]["problem_id"]
    algo_id = config_data["protocol"]["algo_id"]
    config_data["logging"].setdefault("results_base_dir", os.path.join("./results", str(problem_id)))

    module_path = config.get("evalutor_path")
    module = __import__(module_path, fromlist=["RewardingSystem", "generate_initial_population"])
    RewardingSystem = getattr(module, "RewardingSystem")
    generate_initial_population = getattr(module, "generate_initial_population")

    eval_logger = None
    evalutor_path = config.get("evalutor_path")
    project_path = config.get("vmec.project_path")
    maybe_reset_vmec_inputs(evalutor_path, project_path, "pre")

    try:
        property_list = config.get("goals")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        run_name = config.get("exper_name", "run")
        run_id = f"{run_name}_seed{seed}_{ts}"
        run_dir = os.path.join(config_data["logging"]["results_base_dir"], algo_id, run_id)
        eval_logger = EvalLogger(run_dir, problem_id, algo_id, seed, run_id, config_data, property_list)

        reward_system = RewardingSystem(config)
        item_factory = ItemFactory(property_list)

        pop_size = int(config.get("optimization.pop_size"))
        eval_budget = int(config.get("optimization.eval_budget"))

        initial_population_strs = generate_initial_population(config, seed)
        if not initial_population_strs:
            raise RuntimeError("Initial population generator returned no candidates.")

        if len(initial_population_strs) < pop_size:
            initial_population_strs = (list(initial_population_strs) * ((pop_size // len(initial_population_strs)) + 1))[:pop_size]
        else:
            initial_population_strs = list(initial_population_strs)[:pop_size]

        coeff_key_pool = sorted(
            {
                key
                for cand_str in initial_population_strs
                for key in json.loads(cand_str).get("new_coefficients", {}).keys()
            }
        )

        mutation_factor_range = config.get("baseline.mutation_factor_range", [0.85, 1.15])

        initial_sampling = np.array(initial_population_strs, dtype=object).reshape(pop_size, 1)

        problem = MOLLMProblem(reward_system, config, item_factory, eval_logger=eval_logger, pop_size=pop_size)

        algorithm = NSGA2(
            pop_size=int(pop_size),
            sampling=initial_sampling,
            crossover=JSONUniformCrossover(coeff_key_pool),
            mutation=JSONScalingMutation(coeff_key_pool, mutation_factor_range),
            eliminate_duplicates=False,
        )

        termination = ("n_eval", int(eval_budget))

        minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=True,
            save_history=False,
        )
    finally:
        try:
            if eval_logger is not None:
                eval_logger.close()
        except Exception:
            pass
        maybe_reset_vmec_inputs(evalutor_path, project_path, "post")


if __name__ == "__main__":
    main()
