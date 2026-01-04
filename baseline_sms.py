import sys
import os
import argparse
import random
import json
import copy
import numpy as np
from typing import List, Tuple
import pygmo as pg

# ----------------- 路径修正代码 [开始] -----------------
project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------- 路径修正代码 [结束] -----------------

from model.MOLLM import MOLLM, ConfigLoader
# 导入的是我们修改过的 run_ga_baseline，因此 BaselineMOO 已经包含了所有增强功能
from baseline_ga import BaselineMOO  

def _local_corrected_hvc_selection(pops: List, pop_size: int, ref_point: np.ndarray) -> List:
    if not pops: return []
    valid_pops = [p for p in pops if hasattr(p, 'scores') and p.scores is not None]
    if len(valid_pops) <= pop_size: return valid_pops
    scores = np.array([p.scores for p in valid_pops])
    hv_pygmo = pg.hypervolume(scores)
    hvc = hv_pygmo.contributions(ref_point)
    sorted_indices = np.argsort(hvc)[::-1]
    bestn = [valid_pops[i] for i in sorted_indices[:pop_size]]
    return bestn


class SMSEMOABaselineMOO(BaselineMOO):
    """
    SMSEMOA 基线类。
    该类继承自增强后的 BaselineMOO，自动获得强探索能力。
    它唯一的作用是重写种群选择方法，以采用 SMSEMOA 策略。
    """
    def __init__(self, reward_system, llm, property_list, config, seed):
        super().__init__(reward_system, llm, property_list, config, seed)
        print("--- SMSEMOA Baseline MOO has been activated (inherits ENHANCED BaselineMOO) ---")

    def select_next_population(self, pop_size: int) -> List:
        whole_population = [item[0] for item in self.mol_buffer if item[0].total is not None]
        if not whole_population or len(whole_population) <= pop_size: return whole_population
        # print(f"[SMSEMOA Selection] Selecting {pop_size} from an archive of {len(whole_population)}.")
        directions = self.config.get('optimization_direction')
        original_scores = np.array([p.scores for p in whole_population])
        min_vals, max_vals = np.min(original_scores, axis=0), np.max(original_scores, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized_scores = (original_scores - min_vals) / range_vals
        for i, direction in enumerate(directions):
            if direction == 'max': normalized_scores[:, i] = 1.0 - normalized_scores[:, i]
        temp_pop_for_selection = []
        for i, p in enumerate(whole_population):
            temp_p = copy.copy(p)
            temp_p.scores = normalized_scores[i]
            temp_pop_for_selection.append(temp_p)
        ref_point = np.full(original_scores.shape[1], 1.1)
        selected_temp_pops = _local_corrected_hvc_selection(temp_pop_for_selection, pop_size, ref_point)
        selected_values = {p.value for p in selected_temp_pops}
        final_selection = [p for p in whole_population if p.value in selected_values]
        return final_selection


class SMSEMOABaselineRunner(MOLLM):
    def run(self):
        if self.resume: self.load_from_pkl(self.save_path)
        moo = SMSEMOABaselineMOO(self.reward_system, self.llm, self.property_list, self.config, self.seed)
        init_pops, final_pops = moo.run()
        self.history.append(moo.history)
        self.final_pops.append(final_pops)
        self.init_pops.append(init_pops)
        print("\nSMSEMOA baseline run finished.")


def main():
    parser = argparse.ArgumentParser(description='Run SMSEMOA Baseline (config-driven).')
    parser.add_argument('config', type=str, nargs='?', default='sacs_geo_jk/config.yaml', help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=41, help='Random seed for reproducibility')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    cfg_path = args.config
    if cfg_path.startswith('problem/'): cfg_path = cfg_path[len('problem/'):]
    config = ConfigLoader(cfg_path)
    original_suffix = config.get('save_suffix', 'sacs_block_gen')
    config.config['save_suffix'] = f"{original_suffix}_baseline_SMSEMOA"
    print(f"Results will be saved with suffix: {config.get('save_suffix')}")
    print(f"Eval budget (optimization.eval_budget) = {config.get('optimization.eval_budget')}")
    baseline_runner = SMSEMOABaselineRunner(config=config, resume=args.resume, eval=False, seed=args.seed)
    baseline_runner.run()


if __name__ == "__main__":
    main()