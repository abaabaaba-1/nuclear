import numpy as np
import pickle
import os
import json
import time
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
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
        # Extrapolate the last known score to the end of the budget
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
        self.constraints = results.get('constraint_results', {}).get('max_uc', 999.0)

# =========================================================================================================
# Pymoo Problem Definition
# =========================================================================================================

class SACSProblem(Problem):
    """pymoo Problem wrapper for the SACS optimization task."""
    def __init__(self, reward_system, config, item_factory):
        self.reward_system = reward_system
        self.config = config
        self.item_factory = item_factory
        self.history_buffer = [] # To store (Item, eval_count) tuples
        self.eval_count = 0

        n_obj = len(config.get('goals'))
        
        super().__init__(n_vars=1, n_obj=n_obj, n_constr=1, elementwise_evaluation=True, type_vars=object)

    def _evaluate(self, x, out, *args, **kwargs):
        candidate_json_str = x[0]
        item = self.item_factory.create(candidate_json_str)

        evaluated_items, _ = self.reward_system.evaluate([item])
        evaluated_item = evaluated_items[0]

        self.eval_count += 1
        self.history_buffer.append((evaluated_item, self.eval_count))

        out["F"] = np.array(evaluated_item.scores, dtype=float)
        max_uc = evaluated_item.constraints if evaluated_item.constraints is not None else 999.0
        out["G"] = [max_uc - 1.0]

# =========================================================================================================
# Main Runner
# =========================================================================================================

def main():
    # --- 1. Load Configuration ---
    config_path = 'problem/sacs/config.yaml'
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
    seed = 42

    # --- 2. Setup Environment ---
    property_list = config.get('goals')
    save_dir = config.get('save_dir')
    model_name = "baseline_sms_moea"
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
    initial_items = [item_factory.create(s) for s in initial_population_strs]
    
    evaluated_initial_items, _ = reward_system.evaluate(initial_items)
    for i, item in enumerate(evaluated_initial_items):
        item.eval_count = i + 1

    # --- 4. Setup Pymoo Algorithm ---
    problem = SACSProblem(reward_system, config, item_factory)
    problem.history_buffer.extend([(item, item.eval_count) for item in evaluated_initial_items])
    problem.eval_count = len(evaluated_initial_items)

    pop_size = config.get('optimization.pop_size')
    eval_budget = config.get('optimization.eval_budget')

    initial_sampling = np.array(initial_population_strs).reshape(-1, 1)

    algorithm = SMSEMOA(
        pop_size=pop_size,
        sampling=initial_sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20)
    )

    termination = ('n_eval', eval_budget)
    
    # --- 5. Run Optimization ---
    print(f"Running SMS-EMOA for {eval_budget} evaluations...")
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
    final_population_items = [item for item, _ in all_evaluated_items if item.value in [r[0] for r in res.X]]

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
            'top1_auc': top_auc(subset_buffer, 1, False, log_freq, eval_budget),
            'top10_auc': top_auc(subset_buffer, 10, False, log_freq, eval_budget),
            'top100_auc': top_auc(subset_buffer, 100, False, log_freq, eval_budget),
            'running_time[s]': running_time * (i / eval_budget)
        })

    final_top100 = sorted([item[0] for item in all_evaluated_items], key=lambda x: x.total, reverse=True)[:100]
    results_log['results'].append({
        'Training_step': len(all_evaluated_items),
        'all_unique_moles': len(all_evaluated_items),
        'avg_top1': final_top100[0].total if final_top100 else -1,
        'avg_top10': np.mean([item.total for item in final_top100[:10]]) if final_top100 else -1,
        'avg_top100': np.mean([item.total for item in final_top100]) if final_top100 else -1,
        'top1_auc': top_auc(all_evaluated_items, 1, True, log_freq, eval_budget),
        'top10_auc': top_auc(all_evaluated_items, 10, True, log_freq, eval_budget),
        'top100_auc': top_auc(all_evaluated_items, 100, True, log_freq, eval_budget),
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