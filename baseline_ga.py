import sys
import os
import argparse
import random
import json
import copy
import numpy as np
from typing import List, Tuple

# ----------------- 路径修正代码 [开始] -----------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------- 路径修正代码 [结束] -----------------

from model.MOLLM import MOLLM, ConfigLoader
from algorithm.MOO import MOO
from algorithm.base import Item
from problem.sacs_geo_jk.evaluator import _parse_and_modify_line


class BaselineMOO(MOO):
    """
    基线多目标优化类：使用经过优化的经典遗传算子。
    集成了锦标赛选择、自适应变异率和增强的变异强度。
    """

    def __init__(self, reward_system, llm, property_list, config, seed):
        super().__init__(reward_system, llm, property_list, config, seed)
        self.mutation_prob = config.get('baseline.mutation_prob', 0.2)
        self.crossover_prob = config.get('baseline.crossover_prob', 0.8)
        print("--- BaselineMOO in an Optimized Classic Genetic Algorithm mode is activated ---")
        print(f"Base Mutation probability: {self.mutation_prob}, Crossover probability: {self.crossover_prob}")

        # ========== [新功能] 自适应变异率状态变量 ==========
        self.adaptive_mutation_prob = self.mutation_prob
        print(f"Adaptive mutation enabled. Initial rate: {self.adaptive_mutation_prob:.2f}")

        # ========== 基线独立早停配置 ==========
        self.baseline_es_cfg = config.get('baseline_early_stopping', {}) or {}

        def _es_get(k, default):
            return self.baseline_es_cfg.get(k, default)

        # 是否启用
        self.baseline_es_enable = _es_get('enable', True)
        # 选用指标：avg_top100 | hypervolume | top1
        self.baseline_es_metric = _es_get('metric', 'avg_top100')
        # 绝对改进阈值
        self.baseline_es_abs_tol = _es_get('abs_tol', 1e-4)
        # 相对改进阈值
        self.baseline_es_rel_tol = _es_get('rel_tol', 1e-3)
        # 连续停滞轮数
        self.baseline_es_patience = _es_get('patience', 6)
        # 至少多少代后才开始判定
        self.baseline_es_min_gen = _es_get('min_generations', 8)
        # 至少多少唯一评估样本后开始判定
        self.baseline_es_min_samples = _es_get('min_samples', 600)
        # old_score 必须大于该值才计入停滞逻辑
        self.baseline_es_min_score = _es_get('min_score', 0.05)
        # 是否在大跳跃后重置耐心
        self.baseline_es_reset_jump = _es_get('reset_on_jump', True)
        # 大跳跃因子：rel_delta > big_jump_factor * rel_tol
        self.baseline_es_jump_factor = _es_get('big_jump_factor', 5.0)
        # 打印调试信息
        self.baseline_es_verbose = _es_get('verbose', True)
        # 是否覆盖/关闭父类早停
        self.baseline_es_override = _es_get('override_builtin', True)

        # 状态变量
        self.baseline_es_old_score = None
        self.baseline_es_pat_count = 0

        if self.baseline_es_override:
            # 禁用父类默认早停逻辑，使 early_stopping 只受当前判定控制
            self.config.config['early_stopping'] = False
            print("[BaselineES] Builtin early_stopping disabled (override_builtin=True)")

        print("[BaselineES] Config:",
              f"enable={self.baseline_es_enable}, metric={self.baseline_es_metric}, abs_tol={self.baseline_es_abs_tol}, "
              f"rel_tol={self.baseline_es_rel_tol}, patience={self.baseline_es_patience}, "
              f"min_gen={self.baseline_es_min_gen}, min_samples={self.baseline_es_min_samples}")

    # ========== [新功能] 父代锦标赛选择 ==========
    def tournament_selection(self, population: list, k: int = 3) -> Item:
        """
        通过锦标赛选择法从种群中选择一个个体。
        Args:
            population (list): Item对象的列表。
            k (int): 锦标赛的大小。
        Returns:
            Item: 获胜的个体。
        """
        # 从种群中随机选择k个个体
        tournament_contenders = random.sample(population, k)
        
        # 获胜者是总分最高的个体
        winner = max(tournament_contenders, key=lambda item: item.total)
        return winner

    # ========== 基线遗传算子 (已修改，支持 SACS 和 Stellarator_VMEC 两种格式) ==========
    def baseline_genetic_operator(self, parent_list: list) -> tuple:
        parent1, parent2 = parent_list[0], parent_list[1]

        try:
            design1 = json.loads(parent1.value)
            design2 = json.loads(parent2.value)
        except json.JSONDecodeError:
            print("Warning: Failed to decode parent JSON. Skipping operation.")
            return [copy.deepcopy(parent1), copy.deepcopy(parent2)], None, None

        # 自动检测格式：SACS 使用 "new_code_blocks"，Stellarator_VMEC 使用 "new_coefficients"
        if "new_code_blocks" in design1:
            data_key = "new_code_blocks"
            is_sacs_format = True
        elif "new_coefficients" in design1:
            data_key = "new_coefficients"
            is_sacs_format = False
        else:
            print(f"Warning: Unknown design format. Expected 'new_code_blocks' or 'new_coefficients', got keys: {list(design1.keys())}")
            return [copy.deepcopy(parent1), copy.deepcopy(parent2)], None, None

        offspring_design1 = {data_key: {}}
        offspring_design2 = {data_key: {}}

        all_keys = list(design1[data_key].keys())

        # 交叉：基因从父代随机继承 (Uniform Crossover)
        for key in all_keys:
            if random.random() < 0.5:
                offspring_design1[data_key][key] = design1[data_key][key]
                offspring_design2[data_key][key] = design2[data_key][key]
            else:
                offspring_design1[data_key][key] = design2[data_key][key]
                offspring_design2[data_key][key] = design1[data_key][key]

        # 变异 (使用自适应变异率和增强的强度)
        for offspring_design in [offspring_design1, offspring_design2]:
            # [修改] 使用自适应变异率
            if random.random() < self.adaptive_mutation_prob:
                # [修改] 增加变异强度，从10%提升到25%，以加强探索
                num_mutations = random.randint(1, max(1, len(all_keys) // 4))
                keys_to_mutate = random.sample(all_keys, num_mutations)
                for key in keys_to_mutate:
                    if is_sacs_format:
                        # SACS 格式：使用 _parse_and_modify_line 进行字符串变异
                        original_line = offspring_design[data_key][key]
                        block_name = key.replace("_", " ")
                        mutated_line = _parse_and_modify_line(original_line, block_name)
                        offspring_design[data_key][key] = mutated_line
                    else:
                        # Stellarator_VMEC 格式：对系数进行数值变异（2% 扰动）
                        original_value = offspring_design[data_key][key]
                        mutation_factor = random.uniform(0.98, 1.02)
                        offspring_design[data_key][key] = original_value * mutation_factor

        offspring1_str = json.dumps(offspring_design1)
        offspring2_str = json.dumps(offspring_design2)

        new_items = [self.item_factory.create(offspring1_str), self.item_factory.create(offspring2_str)]
        return new_items, None, None

    # ========== 重写生成后代 (已修改) ==========
    def generate_offspring(self, population: list, offspring_times: int) -> list:
        tmp_offspring = []

        for _ in range(offspring_times):
            # [修改] 使用锦标赛选择来挑选具有选择压力的父代
            parent1 = self.tournament_selection(population, k=3)
            parent2 = self.tournament_selection(population, k=3)

            # 确保父代是不同的个体以增加多样性
            while parent1.value == parent2.value:
                parent2 = self.tournament_selection(population, k=3)
            
            child_pair, _, _ = self.baseline_genetic_operator([parent1, parent2])
            tmp_offspring.extend(child_pair)

        self.generated_num += len(tmp_offspring)

        if len(tmp_offspring) == 0:
            return []

        offspring = self.evaluate(tmp_offspring)

        # 保持历史接口结构
        prompts = [None] * offspring_times
        responses = [None] * offspring_times
        generations = [tmp_offspring[i:i + 2] for i in range(0, len(tmp_offspring), 2)]
        self.history.push(prompts, generations, responses)

        return offspring

    def update_experience(self):
        # 基线不做 LLM 经验总结
        pass

    # ========== 重写 log_results (已修改，加入自适应变异逻辑) ==========
    def log_results(self, mol_buffer=None, buffer_type="default", finish=False):
        # 1. 先执行父类逻辑
        super().log_results(mol_buffer=mol_buffer, buffer_type=buffer_type, finish=finish)

        # 2. 仅在 default buffer、非最终写入、启用基线早停时执行
        if buffer_type != "default" or finish or (not self.baseline_es_enable):
            return

        # 冷启动条件判断
        current_gen = self.time_step
        current_samples = len(self.mol_buffer)
        if current_gen < self.baseline_es_min_gen or current_samples < self.baseline_es_min_samples:
            return

        # 取最近一次记录
        if not self.results_dict.get('results'):
            return
        last_record = self.results_dict['results'][-1]

        # 选指标
        metric_name = self.baseline_es_metric
        new_score = last_record.get(metric_name)

        if new_score is None:
            if self.baseline_es_verbose:
                print(f"[BaselineES] metric {metric_name} not available in record.")
            return

        if self.baseline_es_old_score is None:
            # 第一次初始化
            self.baseline_es_old_score = new_score
            return

        delta = new_score - self.baseline_es_old_score
        rel_delta = delta / max(abs(self.baseline_es_old_score), 1e-8)

        stagnate_abs = (abs(delta) < self.baseline_es_abs_tol)
        stagnate_rel = (abs(rel_delta) < self.baseline_es_rel_tol)

        progressed = True
        if self.baseline_es_old_score > self.baseline_es_min_score and (stagnate_abs or stagnate_rel):
            # 停滞
            self.baseline_es_pat_count += 1
            progressed = False
        else:
            # 有改进或未达门槛
            if self.baseline_es_reset_jump and rel_delta > self.baseline_es_jump_factor * self.baseline_es_rel_tol:
                self.baseline_es_pat_count = 0
            else:
                self.baseline_es_pat_count = 0

        # ========== [新功能] 自适应变异率调整 ==========
        if not progressed:
            # 停滞时，提高变异率以增加探索
            self.adaptive_mutation_prob = min(self.mutation_prob * 2.0, 0.9) # 加倍，但最高不超过0.9
            if self.baseline_es_verbose:
                print(f"[BaselineES] Stagnation detected. Increasing mutation probability to {self.adaptive_mutation_prob:.2f}")
        else:
            # 有进展时，恢复到基础变异率
            if abs(self.adaptive_mutation_prob - self.mutation_prob) > 1e-5:
                 if self.baseline_es_verbose:
                    print(f"[BaselineES] Progress detected. Resetting mutation probability to {self.mutation_prob:.2f}")
                 self.adaptive_mutation_prob = self.mutation_prob
        # ===============================================

        if self.baseline_es_verbose:
            print(
                f"[BaselineES] metric={metric_name} new={new_score:.6f} old={self.baseline_es_old_score:.6f} "
                f"Δ={delta:.3e} relΔ={rel_delta:.3e} "
                f"{'STAG' if not progressed else 'OK '} "
                f"pat={self.baseline_es_pat_count}/{self.baseline_es_patience}"
            )

        # 更新历史值
        self.baseline_es_old_score = new_score

        # 触发早停
        if self.baseline_es_pat_count >= self.baseline_es_patience:
            print("[BaselineES] Early stopping triggered (baseline criteria).")
            self.early_stopping = True


class BaselineMOLLM(MOLLM):
    """
    顶层控制器：实例化基线 MOO 并运行
    """
    def run(self):
        if self.resume:
            self.load_from_pkl(self.save_path)

        moo = BaselineMOO(self.reward_system, self.llm, self.property_list, self.config, self.seed)
        init_pops, final_pops = moo.run()

        self.history.append(moo.history)
        self.final_pops.append(final_pops)
        self.init_pops.append(init_pops)

        print("\nOptimized Baseline GA run finished.")


def main():
    parser = argparse.ArgumentParser(description='Run Baseline GA for SACS projects (config-driven).')
    parser.add_argument('config', type=str, nargs='?', default='sacs_geo_jk/config.yaml',
                        help='Path to the configuration file (e.g., config.yaml)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()

    # 载入配置
    # 兼容：框架内部会自动在前面加 problem/，因此如果用户传入已含 problem/，这里先去重一次
    cfg_path = args.config
    if cfg_path.startswith('problem/'): cfg_path = cfg_path[len('problem/'):]
    config = ConfigLoader(cfg_path)

    # 修改保存后缀，防止覆盖 LLM 版本
    original_suffix = config.get('save_suffix', 'sacs_block_gen')
    config.config['save_suffix'] = f"{original_suffix}_baseline_GA_optimized"

    # 基线参数默认
    if 'baseline' not in config.config:
        config.config['baseline'] = {
            'mutation_prob': 0.3,
            'crossover_prob': 0.7
        }

    if 'baseline_early_stopping' not in config.config:
        config.config['baseline_early_stopping'] = {
            'enable': True,
            'metric': 'avg_top100',
            'abs_tol': 1e-4,
            'rel_tol': 1e-3,
            'patience': 6,
            'min_generations': 8,
            'min_samples': 600,
            'min_score': 0.05,
            'reset_on_jump': True,
            'big_jump_factor': 5.0,
            'verbose': True,
            'override_builtin': True
        }

    print(f"Results will be saved with suffix: {config.get('save_suffix')}")
    print(f"Eval budget (optimization.eval_budget) = {config.get('optimization.eval_budget')}")

    # 实例化运行
    baseline_runner = BaselineMOLLM(config=config, resume=args.resume, eval=False, seed=args.seed)
    baseline_runner.run()


if __name__ == "__main__":
    main()