import sys
import os
import argparse
import random
import json
import copy
import numpy as np
import importlib
from typing import List, Tuple

# Optional VMEC reset helper (no-op for other problems)
try:
    from problem.stellarator_vmec.vmec_reset_helper import maybe_reset_vmec_inputs
except ImportError:
    def maybe_reset_vmec_inputs(*_args, **_kwargs):
        return

# ----------------- 路径修正代码 [开始] -----------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------- 路径修正代码 [结束] -----------------

from model.MOLLM import MOLLM, ConfigLoader
from algorithm.MOO import MOO
from algorithm.base import Item

MUTATOR_CACHE = {}


class BaselineMOO(MOO):
    """
    基线多目标优化类：使用经过优化的经典遗传算子。
    集成了锦标赛选择、自适应变异率和增强的变异强度。
    """

    def __init__(self, reward_system, llm, property_list, config, seed):
        super().__init__(reward_system, llm, property_list, config, seed)
        self.mutation_prob = config.get('baseline.mutation_prob', 0.2)
        self.crossover_prob = config.get('baseline.crossover_prob', 0.8)
        self.mutation_factor_range = config.get('baseline.mutation_factor_range', [0.85, 1.15])
        
        # VMEC特定的变异限制（从config读取，与evaluator保持一致）
        self.vmec_low_order_limit = config.get('llm_constraints.low_order_max_rel_change', 0.03)
        self.vmec_high_order_limit = config.get('llm_constraints.high_order_max_rel_change', 0.08)
        
        print("--- BaselineMOO in an Optimized Classic Genetic Algorithm mode is activated ---")
        print(f"Base Mutation probability: {self.mutation_prob}, Crossover probability: {self.crossover_prob}")
        print(f"VMEC mutation limits: low_order=±{self.vmec_low_order_limit*100:.1f}%, high_order=±{self.vmec_high_order_limit*100:.1f}%")

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
        eval_budget = config.get('optimization.eval_budget', 5000)
        self.baseline_es_min_samples = _es_get('min_samples', eval_budget)
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
        self.is_coil_problem = (config.get('evalutor_path') == 'problem.stellarator_coil.evaluator')

    # ========== VMEC系数解析辅助函数 ==========
    def _parse_vmec_mode_numbers(self, key: str):
        """解析VMEC系数键，返回(m, n)模式数字"""
        import re
        pattern = re.compile(r"([RZ]B[CS])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)
        norm_key = key.strip().replace(" ", "")
        match = pattern.match(norm_key)
        if not match:
            return None, None
        return abs(int(match.group(2))), abs(int(match.group(3)))
    
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
        # 处理种群过小或为空的情况
        if not population:
            raise ValueError("Population is empty in tournament_selection.")
        if k > len(population):
            k = len(population)

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

        parent1_blocks = design1.get(data_key, {})
        parent2_blocks = design2.get(data_key, {})
        all_keys = sorted(set(parent1_blocks.keys()) | set(parent2_blocks.keys()))

        # 交叉：基因从父代随机继承 (Uniform Crossover)，允许父代缺少某些键
        for key in all_keys:
            gene1 = parent1_blocks.get(key)
            gene2 = parent2_blocks.get(key)
            if gene1 is None and gene2 is None:
                # 两个父代都缺失该基因，跳过
                continue
            if gene1 is None:
                gene1 = gene2
            if gene2 is None:
                gene2 = gene1

            if random.random() < 0.5:
                offspring_design1[data_key][key] = gene1
                offspring_design2[data_key][key] = gene2
            else:
                offspring_design1[data_key][key] = gene2
                offspring_design2[data_key][key] = gene1

        # 变异 (使用自适应变异率和增强的强度)
        for offspring_design in [offspring_design1, offspring_design2]:
            # [修改] 使用自适应变异率
            if random.random() < self.adaptive_mutation_prob:
                num_mutations = random.randint(1, max(1, len(all_keys) // 2))
                keys_to_mutate = random.sample(all_keys, num_mutations)
                for key in keys_to_mutate:
                    if is_sacs_format:
                        block_name = key.replace("_", " ")
                        mutator = self._get_block_mutator(block_name)
                        original_line = offspring_design[data_key][key]
                        mutated_line = mutator(original_line, block_name)
                        offspring_design[data_key][key] = mutated_line
                    else:
                        # VMEC系数变异：根据系数阶数使用不同的变异限制
                        original_value = offspring_design[data_key].get(key)
                        if original_value is None or original_value == 0.0:
                            continue
                        
                        # 判断是低阶还是高阶系数
                        m, n = self._parse_vmec_mode_numbers(key)
                        if m is not None and n is not None and m <= 2 and n <= 1:
                            # 低阶系数：±3%
                            rel_limit = self.vmec_low_order_limit
                        else:
                            # 高阶系数：±8%
                            rel_limit = self.vmec_high_order_limit
                        
                        # 应用变异
                        mutation_delta = random.uniform(-rel_limit, rel_limit)
                        offspring_design[data_key][key] = original_value * (1.0 + mutation_delta)

        offspring1_str = json.dumps(offspring_design1)
        offspring2_str = json.dumps(offspring_design2)

        new_items = [self.item_factory.create(offspring1_str), self.item_factory.create(offspring2_str)]
        return new_items, None, None

    def coil_genetic_operator(self, parent_list: list) -> tuple:
        """
        闭环基遗传算子：支持新的 Loop Basis 格式
        操作对象：{"loops": [{"loop_id": int, "current": float}, ...]}
        """
        parent1, parent2 = parent_list[0], parent_list[1]
        try:
            design1 = json.loads(parent1.value)
            design2 = json.loads(parent2.value)
        except json.JSONDecodeError:
            return [copy.deepcopy(parent1), copy.deepcopy(parent2)], None, None
        
        if not isinstance(design1, dict) or not isinstance(design2, dict):
            return [copy.deepcopy(parent1), copy.deepcopy(parent2)], None, None
        
        # 支持闭环基格式（新）
        if 'loops' in design1 and 'loops' in design2:
            loops1 = design1.get('loops', []) or []
            loops2 = design2.get('loops', []) or []
            
            # 转换为字典便于操作：{loop_id: current}
            loops1_dict = {loop['loop_id']: loop['current'] for loop in loops1 if 'loop_id' in loop and 'current' in loop}
            loops2_dict = {loop['loop_id']: loop['current'] for loop in loops2 if 'loop_id' in loop and 'current' in loop}
            
            # 交叉：随机组合父代的闭环
            all_loop_ids = sorted(set(loops1_dict.keys()) | set(loops2_dict.keys()))
            child1_loops = {}
            child2_loops = {}
            
            for loop_id in all_loop_ids:
                has1 = loop_id in loops1_dict
                has2 = loop_id in loops2_dict
                
                if has1 and has2:
                    # 两个父代都有这个闭环，随机分配
                    if random.random() < 0.5:
                        child1_loops[loop_id] = loops1_dict[loop_id]
                        child2_loops[loop_id] = loops2_dict[loop_id]
                    else:
                        child1_loops[loop_id] = loops2_dict[loop_id]
                        child2_loops[loop_id] = loops1_dict[loop_id]
                elif has1:
                    # 只有父代1有，随机决定哪个子代继承
                    if random.random() < 0.5:
                        child1_loops[loop_id] = loops1_dict[loop_id]
                    else:
                        child2_loops[loop_id] = loops1_dict[loop_id]
                else:
                    # 只有父代2有
                    if random.random() < 0.5:
                        child1_loops[loop_id] = loops2_dict[loop_id]
                    else:
                        child2_loops[loop_id] = loops2_dict[loop_id]
            
            # 变异
            min_I = self.config.get('coil_design.min_current', 0.1)
            max_I = self.config.get('coil_design.max_current', 2.0)
            low_factor, high_factor = self.mutation_factor_range
            
            def mutate_loops(loops_dict):
                """对闭环进行变异"""
                if not loops_dict or random.random() >= self.adaptive_mutation_prob:
                    return loops_dict
                
                mutated = loops_dict.copy()
                loop_ids = list(mutated.keys())
                
                # 变异操作：
                # 1. 修改电流 (60%)
                # 2. 删除闭环 (20%)
                # 3. 添加新闭环 (20%)
                
                mutation_type = random.random()
                
                if mutation_type < 0.6 and len(loop_ids) > 0:
                    # 修改电流
                    num_mut = random.randint(1, max(1, len(loop_ids) // 3))
                    for loop_id in random.sample(loop_ids, min(num_mut, len(loop_ids))):
                        current = mutated[loop_id]
                        if random.random() < 0.5:
                            # 缩放
                            factor = random.uniform(low_factor, high_factor)
                            new_current = current * factor
                        else:
                            # 翻转
                            new_current = -current
                        
                        # 限制范围
                        new_current = max(min(new_current, max_I), -max_I)
                        if abs(new_current) < min_I:
                            new_current = min_I if new_current >= 0 else -min_I
                        
                        mutated[loop_id] = new_current
                
                elif mutation_type < 0.8 and len(loop_ids) > 3:
                    # 删除闭环（保持至少3个）
                    num_remove = random.randint(1, max(1, len(loop_ids) // 5))
                    for loop_id in random.sample(loop_ids, min(num_remove, len(loop_ids) - 3)):
                        del mutated[loop_id]
                
                else:
                    # 添加新闭环（从闭环库中随机选择）
                    # 假设闭环库有约42000个闭环
                    num_add = random.randint(1, 3)
                    for _ in range(num_add):
                        new_loop_id = random.randint(0, 41999)  # 闭环库范围
                        new_current = random.uniform(min_I, max_I) * random.choice([-1, 1])
                        mutated[new_loop_id] = new_current
                
                return mutated
            
            child1_loops = mutate_loops(child1_loops)
            child2_loops = mutate_loops(child2_loops)
            
            # 构建输出格式
            def build_loop_design(loops_dict):
                if not loops_dict:
                    # 至少保留一个随机闭环
                    return {"loops": [{"loop_id": random.randint(0, 100), "current": random.uniform(min_I, max_I)}]}
                return {"loops": [{"loop_id": lid, "current": round(cur, 3)} for lid, cur in loops_dict.items()]}
            
            offspring1_str = json.dumps(build_loop_design(child1_loops))
            offspring2_str = json.dumps(build_loop_design(child2_loops))
            new_items = [self.item_factory.create(offspring1_str), self.item_factory.create(offspring2_str)]
            return new_items, None, None
        
        # 旧格式兼容（segment basis）- 保留原有逻辑
        else:
            segs1 = set(design1.get('active_segments', []) or [])
            segs2 = set(design2.get('active_segments', []) or [])
            curr1_raw = design1.get('currents', {}) or {}
            curr2_raw = design2.get('currents', {}) or {}
            curr1 = {}
            for k, v in curr1_raw.items():
                try:
                    idx = int(k)
                    curr1[idx] = float(v)
                except (ValueError, TypeError):
                    continue
            curr2 = {}
            for k, v in curr2_raw.items():
                try:
                    idx = int(k)
                    curr2[idx] = float(v)
                except (ValueError, TypeError):
                    continue
            all_segments = sorted(set(curr1.keys()) | set(curr2.keys()) | segs1 | segs2)
            child1_curr = {}
            child2_curr = {}
            for seg in all_segments:
                has1 = seg in curr1
                has2 = seg in curr2
                if not has1 and not has2:
                    continue
                if has1 and has2:
                    if random.random() < 0.5:
                        v1 = curr1[seg]
                        v2 = curr2[seg]
                    else:
                        v1 = curr2[seg]
                        v2 = curr1[seg]
                elif has1:
                    v1 = curr1[seg]
                    v2 = curr1[seg]
                else:
                    v1 = curr2[seg]
                    v2 = curr2[seg]
                child1_curr[seg] = v1
                child2_curr[seg] = v2
            min_I = self.config.get('coil_design.min_current', default=0.0) or 0.0
            max_I = self.config.get('coil_design.max_current', default=0.0)
            if max_I is None or max_I <= 0.0:
                max_I = 5.0
            low_factor, high_factor = self.mutation_factor_range[0], self.mutation_factor_range[1]
            def mutate(curr_dict):
                if not curr_dict:
                    return curr_dict
                if random.random() >= self.adaptive_mutation_prob:
                    return curr_dict
                seg_list = list(curr_dict.keys())
                num_mut = random.randint(1, max(1, len(seg_list) // 5))
                for seg in random.sample(seg_list, num_mut):
                    val = curr_dict[seg]
                    if random.random() < 0.5:
                        factor = random.uniform(low_factor, high_factor)
                        new_val = val * factor
                    else:
                        new_val = -val
                    amp = abs(new_val)
                    if min_I > 0.0 and amp < min_I:
                        new_val = min_I if new_val >= 0.0 else -min_I
                    if max_I > 0.0 and amp > max_I:
                        new_val = max_I if new_val >= 0.0 else -max_I
                    curr_dict[seg] = new_val
                return curr_dict
            child1_curr = mutate(child1_curr)
            child2_curr = mutate(child2_curr)
            def build_design(curr_dict):
                if not curr_dict:
                    return {"active_segments": [], "currents": {}}
                active = sorted(curr_dict.keys())
                currents_json = {str(k): float(v) for k, v in curr_dict.items()}
                return {"active_segments": active, "currents": currents_json}
            offspring1_str = json.dumps(build_design(child1_curr))
            offspring2_str = json.dumps(build_design(child2_curr))
            new_items = [self.item_factory.create(offspring1_str), self.item_factory.create(offspring2_str)]
            return new_items, None, None

    def _get_block_mutator(self, block_name: str):
        """Fetches a mutation function appropriate for the block prefix via evaluator module."""
        prefix = block_name.split()[0]
        if prefix == 'JOINT':
            key = 'mutator_joint'
            attr_candidates = ['_parse_and_modify_line']
        elif prefix in {'GRUP', 'PGRUP'}:
            key = 'mutator_section'
            attr_candidates = ['_parse_and_modify_section_line', '_parse_and_modify_line']
        elif prefix.startswith('BC') or prefix.startswith('COEF'):
            key = 'mutator_vmec'
            attr_candidates = ['_parse_and_modify_vmec_line', '_parse_and_modify_line']
        else:
            key = f"mutator_{prefix}"
            attr_candidates = ['_parse_and_modify_line']

        if key not in MUTATOR_CACHE:
            module_path = self.config.get('evalutor_path')
            module = importlib.import_module(module_path)
            target = None
            for attr in attr_candidates:
                fn = getattr(module, attr, None)
                if callable(fn):
                    target = fn
                    break
            if target is None:
                target = lambda line, *_: line
            MUTATOR_CACHE[key] = target
        return MUTATOR_CACHE[key]

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
            
            if self.is_coil_problem:
                child_pair, _, _ = self.coil_genetic_operator([parent1, parent2])
            else:
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
        # Fixed: Use num_gen instead of time_step (time_step is only updated in use_au mode)
        # Use getattr with default 0 to avoid AttributeError before MOO.run() initializes num_gen
        current_gen = getattr(self, 'num_gen', 0)
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
        evalutor_path = self.config.get('evalutor_path')
        project_path = self.config.get('vmec.project_path')
        maybe_reset_vmec_inputs(evalutor_path, project_path, 'pre')
        try:
            if self.resume:
                self.load_from_pkl(self.save_path)

            moo = BaselineMOO(self.reward_system, self.llm, self.property_list, self.config, self.seed)
            init_pops, final_pops = moo.run()

            self.history.append(moo.history)
            self.final_pops.append(final_pops)
            self.init_pops.append(init_pops)

            print("\nOptimized Baseline GA run finished.")
        finally:
            maybe_reset_vmec_inputs(evalutor_path, project_path, 'post')


def main():
    parser = argparse.ArgumentParser(description='Run Baseline GA for SACS projects (config-driven).')
    parser.add_argument('config', type=str, nargs='?', default='sacs_geo_jk/config.yaml',
                        help='Path to the configuration file (e.g., config.yaml)')
    parser.add_argument('--seed', type=int, default=41, help='Random seed for reproducibility')
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
