import argparse
import json
import os
import pickle
import random
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
import yaml

# =========================================================================================================
# UTILS (Copied from your framework for consistency)
# =========================================================================================================

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """Calculates the Area Under the Curve for top-N performance."""
    s = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer, key=lambda kv: kv[1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls) + 1, freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[0].total, reverse=True))[:top_n]
        top_n_now = np.mean([item[0].total for item in temp_result]) if temp_result else 0
        s += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    if len(buffer) > called:
        temp_result = list(sorted([item for item in ordered_results if item[1] <= len(buffer)], key=lambda kv: kv[0].total, reverse=True))[:top_n]
        top_n_now = np.mean([item[0].total for item in temp_result]) if temp_result else 0
        s += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        last_score = np.mean([item[0].total for item in sorted(ordered_results, key=lambda kv: kv[0].total, reverse=True)[:top_n]]) if ordered_results else 0
        s += (max_oracle_calls - len(buffer)) * last_score
    return s / max_oracle_calls


class Item:
    """A simple container for a candidate and its scores."""
    def __init__(self, value, property_list):
        self.value = value
        self.property_list = property_list
        self.scores = None
        self.total = None
        self.property = None
        self.constraints = None

    def assign_results(self, results):
        self.property = results.get('original_results', {})
        self.scores = [results.get('transformed_results', {}).get(obj, 1.0) for obj in self.property_list]
        self.total = results.get('overall_score', -1.0)
        
        # 兼容两种约束格式：
        # 1. SACS: 使用 max_uc 标量约束
        # 2. VMEC: 使用 is_feasible 布尔约束（通过 penalty 已反映在 scores 中）
        constraint_results = results.get('constraint_results', {}) or {}
        if 'max_uc' in constraint_results:
            # SACS 格式：max_uc 是实际约束值
            self.constraints = constraint_results.get('max_uc', 999.0)
        elif 'is_feasible' in constraint_results:
            # VMEC 格式：is_feasible 是布尔值，MOEA/D 不使用显式约束
            # 因为 evaluator 已经在 scores 中加入了 penalty
            self.constraints = 0.0 if bool(constraint_results.get('is_feasible', 0)) else 0.0
        else:
            # 未知格式，默认无约束
            self.constraints = 0.0

# =========================================================================================================
# Pymoo Problem Definition
# =========================================================================================================

class SACSProblem(Problem):
    """pymoo Problem wrapper for the SACS optimization task."""
    def __init__(self, reward_system, config, item_factory):
        self.reward_system = reward_system
        self.config = config
        self.item_factory = item_factory
        self.history_buffer = [] 
        self.eval_count = 0
        n_obj = len(config.get('goals'))
        super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=0, elementwise=True, vtype=object)

    def _evaluate(self, x, out, *args, **kwargs):
        candidate_json_str = x[0]
        item = self.item_factory.create(candidate_json_str)
        evaluated_items, _ = self.reward_system.evaluate([item])
        evaluated_item = evaluated_items[0]
        self.eval_count += 1
        self.history_buffer.append((evaluated_item, self.eval_count))
        out["F"] = np.array(evaluated_item.scores, dtype=float)
        # Retain max_uc for logging/analysis but rely on the evaluator's penalty instead of pymoo constraints.

# =========================================================================================================
# Main Runner
# =========================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baseline MOEA/D for MOLLM problems")
    parser.add_argument('config', nargs='?', default='sacs_geo_jk/config.yaml', help='Path to config YAML (e.g., problem/stellarator_vmec/config_moead.yaml)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config_path = args.config
    if not (config_path.startswith('problem/') or os.path.isabs(config_path)):
        config_path = os.path.join('problem', config_path)
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    class Config:
        def __init__(self, data):
            self._data = data
        def get(self, key, default=None):
            keys = key.split('.')
            val = self._data
            try:
                for k in keys:
                    val = val[k]
                return val
            except (KeyError, TypeError):
                return default
        def to_string(self):
            return yaml.dump(self._data)

    config = Config(config_data)
    seed = args.seed

    # --- 2. Setup Environment ---
    property_list = config.get('goals')
    save_dir = config.get('save_dir')
    model_name = "baseline_moead"
    save_suffix = config.get('save_suffix')
    save_dir_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir_path, exist_ok=True)
    
    module_path = config.get('evalutor_path')
    module = __import__(module_path, fromlist=['RewardingSystem', 'generate_initial_population'])
    RewardingSystem = module.RewardingSystem
    generate_initial_population = module.generate_initial_population
    
    reward_system = RewardingSystem(config)
    
    class ItemFactory:
        def __init__(self, property_list):
            self.property_list = property_list
        def create(self, value):
            return Item(value, self.property_list)

    item_factory = ItemFactory(property_list)

    # --- 3. Initialize Population ---
    print("Generating initial population...")
    initial_population_strs = generate_initial_population(config, seed)

    # --- 4. Setup Pymoo Algorithm ---
    problem = SACSProblem(reward_system, config, item_factory)

    pop_size = config.get('optimization.pop_size')
    eval_budget = config.get('optimization.eval_budget')

    initial_sampling = np.array(initial_population_strs, dtype=object).reshape(-1, 1)

    # 收集可用键用于交叉/变异
    coeff_key_pool = sorted({
        key
        for cand_str in initial_population_strs
        for key in json.loads(cand_str).get('new_coefficients', {}).keys()
    })

    mutation_factor_range = config.get('baseline.mutation_factor_range', [0.85, 1.15])

    class JSONUniformCrossover(Crossover):
        def __init__(self, coeff_keys):
            super().__init__(2, 2)
            self.coeff_keys = coeff_keys

        def _do(self, problem, X, **kwargs):
            n_matings = X.shape[1]
            offspring = np.empty((self.n_offsprings, n_matings, problem.n_var), dtype=object)
            for m in range(n_matings):
                parent_a = json.loads(X[0, m, 0])
                parent_b = json.loads(X[1, m, 0])
                child1 = {'new_coefficients': {}}
                child2 = {'new_coefficients': {}}
                keys = set(parent_a.get('new_coefficients', {}).keys()) | set(parent_b.get('new_coefficients', {}).keys())
                if not keys:
                    keys = set(self.coeff_keys)
                for key in keys:
                    val_a = parent_a.get('new_coefficients', {}).get(key)
                    val_b = parent_b.get('new_coefficients', {}).get(key)
                    if val_a is None and val_b is None:
                        continue
                    if random.random() < 0.5:
                        child1['new_coefficients'][key] = val_a if val_a is not None else val_b
                        child2['new_coefficients'][key] = val_b if val_b is not None else val_a
                    else:
                        child1['new_coefficients'][key] = val_b if val_b is not None else val_a
                        child2['new_coefficients'][key] = val_a if val_a is not None else val_b
                offspring[0, m, 0] = json.dumps(child1)
                offspring[1, m, 0] = json.dumps(child2)
            return offspring

    class JSONScalingMutation(Mutation):
        def __init__(self, coeff_keys, factor_range):
            super().__init__()
            self.coeff_keys = coeff_keys
            self.low, self.high = factor_range

        def _do(self, problem, X, **kwargs):
            mutated = np.empty_like(X, dtype=object)
            for idx in range(len(X)):
                design = json.loads(X[idx, 0])
                coeffs = design.setdefault('new_coefficients', {})
                if not coeffs:
                    mutated[idx, 0] = json.dumps(design)
                    continue
                num_mutations = random.randint(1, max(1, len(coeffs)))
                mutate_keys = random.sample(list(coeffs.keys()), num_mutations)
                for key in mutate_keys:
                    coeffs[key] = coeffs[key] * random.uniform(self.low, self.high)
                mutated[idx, 0] = json.dumps(design)
            return mutated

    # Create reference directions
    n_obj = len(config.get('goals'))
    ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
        sampling=initial_sampling,
        crossover=JSONUniformCrossover(coeff_key_pool),
        mutation=JSONScalingMutation(coeff_key_pool, mutation_factor_range)
    )

    termination = ('n_eval', eval_budget)
    
    # --- 5. Run Optimization ---
    print(f"Running MOEA/D for {eval_budget} evaluations...")
    start_time = time.time()
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   verbose=True,
                   save_history=False)

    running_time = time.time() - start_time
    print(f"Optimization finished in {running_time / 3600:.2f} hours.")

    # --- 6. Process and Save Results ---
    all_evaluated_items = problem.history_buffer
    # Capture the first |initial_population| evaluations as the recorded initial population
    evaluated_initial_items = [item for item, _ in all_evaluated_items[:len(initial_population_strs)]]

    # Extract the final population items corresponding to the MOEA/D solution set
    solution_values = {individual[0] for individual in res.X}
    final_population_items = [item for item, _ in all_evaluated_items if item.value in solution_values]

    results_log = { 'results': [], 'params': config.to_string() }
    log_freq = config.get('optimization.log_freq')
    for i in range(log_freq, eval_budget + 1, log_freq):
        subset_buffer = [item for item in all_evaluated_items if item[1] <= i]
        if not subset_buffer: continue
        top100_items = sorted([item[0] for item in subset_buffer], key=lambda x: x.total, reverse=True)[:100]
        results_log['results'].append({
            'Training_step': i,
            'all_unique_moles': len(subset_buffer),
            'avg_top1': top100_items[0].total if top100_items else -1,
            'avg_top10': np.mean([item.total for item in top100_items[:10]]) if top100_items else -1,
            'avg_top100': np.mean([item.total for item in top100_items]) if top100_items else -1,
            'top1_auc': top_auc(subset_buffer, 1, False, 100, eval_budget),
            'top10_auc': top_auc(subset_buffer, 10, False, 100, eval_budget),
            'top100_auc': top_auc(subset_buffer, 100, False, 100, eval_budget),
            'running_time[s]': running_time * (i / eval_budget)
        })

    final_top100 = sorted([item[0] for item in all_evaluated_items], key=lambda x: x.total, reverse=True)[:100]
    results_log['results'].append({
        'Training_step': len(all_evaluated_items),
        'all_unique_moles': len(all_evaluated_items),
        'avg_top1': final_top100[0].total if final_top100 else -1,
        'avg_top10': np.mean([item.total for item in final_top100[:10]]) if final_top100 else -1,
        'avg_top100': np.mean([item.total for item in final_top100]) if final_top100 else -1,
        'top1_auc': top_auc(all_evaluated_items, 1, True, 100, eval_budget),
        'top10_auc': top_auc(all_evaluated_items, 10, True, 100, eval_budget),
        'top100_auc': top_auc(all_evaluated_items, 100, True, 100, eval_budget),
        'running_time[s]': running_time
    })
    
    json_path = os.path.join(save_dir_path, f"{'_'.join(property_list)}_{save_suffix}_{seed}.json")
    with open(json_path, 'w') as f:
        json.dump(results_log, f, indent=4)
    print(f"JSON results saved to {json_path}")

    pkl_path = os.path.join(save_dir_path, f"{'_'.join(property_list)}_{save_suffix}.pkl")
    data_to_save = {
        'init_pops': evaluated_initial_items,
        'final_pops': final_population_items,
        'all_mols': all_evaluated_items,
        'properties': property_list,
        'evaluation': results_log['results'],
        'running_time': f'{running_time / 3600:.2f} hours'
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"PKL data saved to {pkl_path}")

if __name__ == "__main__":
    main()
