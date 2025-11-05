# problem/stellarator_vmec/evaluator.py (已修正)
import numpy as np
import json
import logging
import random
import copy
import re
import netCDF4
import vmecpp # 确保 vmecpp 已经安装在您的 Conda 环境中 (pip install vmecpp)
from pathlib import Path
from typing import Dict, List, Any

# 从您的框架导入
from .vmec_file_modifier import VmecFileModifier

# --- 1. 种子定义 (从 input.w7x 提取的基准系数) ---
# 注意：这是一个简短的示例。在实际使用中，您应该使用 VmecFileModifier
# 自动从 input.w7x 提取这个字典。
SEED_BASELINE_COEFFS = {
    "RBC(0,0)": 5.5586e+00, "ZBS(0,0)": -0.0000e+00,
    "RBC(1,0)": 2.6447e-01, "ZBS(1,0)": -2.3754e-01,
    "RBC(2,0)": 1.6949e-03, "ZBS(2,0)": -6.8221e-03,
    "RBC(3,0)": -9.4726e-04, "ZBS(3,0)": 3.1833e-03,
    "RBC(4,0)": -1.7618e-03, "ZBS(4,0)": 1.7251e-03,
    "RBC(5,0)": 4.5981e-05, "ZBS(5,0)": 5.7592e-05,
    "RBC(6,0)": -2.3437e-04, "ZBS(6,0)": 2.6430e-04,
    "RBC(-6,1)": 5.8671e-04, "ZBS(-6,1)": 1.3323e-03,
    "RBC(-5,1)": -2.0214e-03, "ZBS(-5,1)": -1.6138e-03,
    "RBC(-4,1)": -3.2499e-03, "ZBS(-4,1)": -4.2108e-03,
    "RBC(-3,1)": -4.8626e-03, "ZBS(-3,1)": -7.0096e-03,
    "RBC(-2,1)": 9.9524e-03, "ZBS(-2,1)": 9.6053e-03,
    "RBC(-1,1)": 3.3555e-02, "ZBS(-1,1)": 3.6669e-02,
    "RBC(0,1)": 4.9093e-01, "ZBS(0,1)": 6.1965e-01,
    "RBC(1,1)": -2.5107e-01, "ZBS(1,1)": 1.7897e-01,
    # ... (为了简洁省略了其余系数) ...
    "RBC(6,6)": 9.7208e-05, "ZBS(6,6)": -1.8940e-04
}

# 包装成框架期望的种子格式
SEED_BASELINE = {"new_coefficients": SEED_BASELINE_COEFFS}

# 初始种子列表（可以添加更多）
INITIAL_SEEDS = [ SEED_BASELINE ]

# --- 2. 初始种群生成 (来自 SACS 评估器的逻辑) ---

def _mutate_seed_coefficients(seed_coeffs: Dict[str, float]) -> Dict[str, float]:
    """对一组系数进行随机突变以生成新个体"""
    mutated_coeffs = copy.deepcopy(seed_coeffs)
    
    # 随机选择 1 到 5 个系数进行突变
    num_to_mutate = random.randint(1, 5)
    keys_to_mutate = random.sample(list(mutated_coeffs.keys()), num_to_mutate)
    
    for key in keys_to_mutate:
        original_value = mutated_coeffs[key]
        
        # (*** 已修正 BUG 4 ***)
        # 施加一个 2% 的随机扰动 (10% 太大了，会导致物理不稳定)
        mutation_factor = random.uniform(0.98, 1.02) # 之前是 (0.9, 1.1)
        
        mutated_coeffs[key] = original_value * mutation_factor
        
    return mutated_coeffs

def generate_initial_population(config, seed):
    np.random.seed(seed)
    random.seed(seed)
    population_size = config.get('optimization.pop_size')
    
    # 实例化一个修饰器来获取基准种子
    # (注意: 确保 config 中的路径正确)
    try:
        base_modifier = VmecFileModifier(
            project_path=config.get('vmec.project_path'),
            input_file=config.get('vmec.input_file')
        )
        base_coeffs = base_modifier.extract_coefficients()
        if not base_coeffs:
            logging.error("无法从 input.w7x 提取基准系数，将使用硬编码的种子。")
            base_coeffs = SEED_BASELINE_COEFFS
    except Exception as e:
        logging.error(f"初始化 VmecFileModifier 失败: {e}。将使用硬编码的种子。")
        base_coeffs = SEED_BASELINE_COEFFS

    initial_seeds = [{"new_coefficients": base_coeffs}]
    
    initial_population_jsons = []
    seen_candidates = set()
    max_tries = population_size * 10

    logging.info(f"正在生成大小为 {population_size} 的初始种群...")
    
    for seed_candidate in initial_seeds:
        candidate_str = json.dumps(seed_candidate, sort_keys=True)
        if candidate_str not in seen_candidates:
            initial_population_jsons.append(candidate_str)
            seen_candidates.add(candidate_str)
    
    try_count = 0
    while len(initial_population_jsons) < population_size and try_count < max_tries:
        base_candidate = copy.deepcopy(random.choice(initial_seeds))
        
        # 应用突变
        mutated_coeffs = _mutate_seed_coefficients(base_candidate["new_coefficients"])
        base_candidate["new_coefficients"] = mutated_coeffs
        
        candidate_str = json.dumps(base_candidate, sort_keys=True)
        if candidate_str not in seen_candidates:
            initial_population_jsons.append(candidate_str)
            seen_candidates.add(candidate_str)
        try_count += 1

    if len(initial_population_jsons) < population_size:
        logging.warning(f"仅生成了 {len(initial_population_jsons)}/{population_size} 个初始候选体。")

    logging.info(f"成功生成 {len(initial_population_jsons)} 个初始候选体。")
    return initial_population_jsons


# --- 3. 结果分析器 (已废弃) ---
# def _analyze_wout_file(filename="wout_w7x.nc"):
#    ... (此函数逻辑已移至 RewardingSystem.evaluate 内部) ...


# --- 4. 核心评估器类 (RewardingSystem) ---

class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.sacs_project_path = config.get('vmec.project_path') # 复用路径
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.input_file = config.get('vmec.input_file')
        
        # (*** 已修改：重新启用 ***) 
        # 恢复 output_file_path 以便手动查验
        self.output_file_path = Path(self.sacs_project_path) / config.get('vmec.output_file')
        
        self.modifier = VmecFileModifier(self.sacs_project_path, self.input_file)
        
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

    def evaluate(self, items):
        invalid_num = 0
        for item in items:
            analysis_res = {} # 存储从 vmec_output 提取的指标
            is_feasible = False
            is_converged = False
            is_stable = False
            min_mercier = -999.0

            try:
                raw_value = item.value
                try:
                    if 'candidate' in raw_value:
                        raw_value = raw_value.split('<candidate>', 1)[1].rsplit('</candidate>', 1)[0].strip()
                    modifications = json.loads(raw_value)
                    new_coefficients = modifications.get("new_coefficients")
                except (json.JSONDecodeError, IndexError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse candidate JSON: {raw_value}. Error: {e}")
                    self._assign_penalty(item, "Invalid JSON format from LLM")
                    invalid_num += 1
                    continue

                if not new_coefficients or not isinstance(new_coefficients, dict):
                    self._assign_penalty(item, "Invalid candidate structure (no new_coefficients)")
                    invalid_num += 1
                    continue
                
                # 1. 修改 input.w7x 文件
                if not self.modifier.replace_coefficients(new_coefficients):
                    self._assign_penalty(item, "VMEC file modification failed")
                    invalid_num += 1
                    continue

                # 2. 运行 VMEC++
                vmec_output = None
                try:
                    # 从修改后的文件加载输入
                    vmec_input = vmecpp.VmecInput.from_file(self.modifier.input_file_path)
                    # 运行计算
                    vmec_output = vmecpp.run(vmec_input)
                    
                    # (*** 已修改：重新启用 ***) 
                    # 将 wout.nc 保存到磁盘，以便手动查验
                    try:
                        vmec_output.wout.save(self.output_file_path)
                        self.logger.info(f"Output file saved for manual inspection: {self.output_file_path}")
                    except Exception as save_e:
                        self.logger.error(f"Failed to save wout.nc file: {save_e}")
                    
                except Exception as e:
                    self.logger.warning(f"VMEC++ 运行失败: {e}")
                    self._assign_penalty(item, f"VMEC_Run_Fail: {str(e)[:100]}")
                    invalid_num += 1
                    continue
                
                # 3. (已修改) 直接从内存中的 vmec_output 分析结果
                try:
                    # 3a. 收敛性 (约束)
                    # 以 VMEC++ 判定为准：fsqr、fsqz、fsql 三者分别 <= ftolv 则收敛
                    tolerance = float(vmec_output.wout.ftolv)
                    fsqr = float(vmec_output.wout.fsqr)
                    fsqz = float(vmec_output.wout.fsqz)
                    fsql = float(vmec_output.wout.fsql)
                    # 记录 fsqt 末值，仅用于参考日志
                    final_residual_sum = float(vmec_output.wout.fsqt[-1]) if getattr(vmec_output.wout, 'fsqt', None) is not None else float(fsqr + fsqz + fsql)
                    is_converged = (fsqr <= tolerance) and (fsqz <= tolerance) and (fsql <= tolerance)
                    analysis_res["is_converged"] = is_converged
                    analysis_res["ftolv"] = tolerance
                    analysis_res["fsqr"] = fsqr
                    analysis_res["fsqz"] = fsqz
                    analysis_res["fsql"] = fsql
                    analysis_res["final_residual_sum"] = final_residual_sum

                    # 3b. 体积 (目标)
                    analysis_res["volume"] = vmec_output.wout.volume_p

                    # 3c. 旋转变换 (目标)
                    iotas_profile = vmec_output.wout.iotas
                    iota_at_axis = iotas_profile[1]
                    iota_at_edge = iotas_profile[-1]
                    magnetic_shear = iota_at_edge - iota_at_axis
                    analysis_res["iota_axis"] = iota_at_axis
                    analysis_res["iota_edge"] = iota_at_edge
                    analysis_res["magnetic_shear"] = magnetic_shear # 目标

                    # 3d. 纵横比 (目标)
                    rmnc = vmec_output.wout.rmnc # 形状 (mnmax, ns) = (288, 99)
                    xm = vmec_output.wout.xm   # 形状 (mnmax,) = (288,)
                    ns = vmec_output.wout.ns   # 99
                    last_surface_idx = ns - 1    # 98
                    
                    # (*** 已修正 BUG 3 - 最终版 ***)
                    # 我们需要第 98 列 (last_surface_idx)
                    R_outer = np.dot(rmnc[:, last_surface_idx], np.cos(xm * 0.0))
                    R_inner = np.dot(rmnc[:, last_surface_idx], np.cos(xm * np.pi))
                    
                    R_major = (R_outer + R_inner) / 2
                    a_minor = (R_outer - R_inner) / 2
                    analysis_res["aspect_ratio"] = R_major / a_minor # 目标

                    # 3e. Mercier 稳定性 (约束)
                    min_mercier = -999.0 # 惩罚值
                    
                    # 优先从 .mercier 对象获取
                    if hasattr(vmec_output, 'mercier') and vmec_output.mercier and hasattr(vmec_output.mercier, 'DShear') and vmec_output.mercier.DShear is not None:
                        min_mercier = np.min(vmec_output.mercier.DShear)
                    else:
                        # 回退到 wout 对象 (以防万一)
                        mercier_data = None
                        if hasattr(vmec_output.wout, 'DShear'):
                            mercier_data = vmec_output.wout.DShear
                        elif hasattr(vmec_output.wout, 'dmier'):
                            mercier_data = vmec_output.wout.dmier
                        
                        if mercier_data is not None and hasattr(mercier_data, 'shape'):
                            min_mercier = np.min(mercier_data)
                        else:
                            # 新增：回退到磁盘 wout.nc 读取
                            try:
                                from netCDF4 import Dataset
                                if self.output_file_path.is_file():
                                    with Dataset(str(self.output_file_path), 'r') as ds:
                                        ds.set_always_mask(False)
                                        if 'DShear' in ds.variables:
                                            min_mercier = float(np.min(ds.variables['DShear'][()]))
                                        elif 'dmier' in ds.variables:
                                            min_mercier = float(np.min(ds.variables['dmier'][()]))
                                        else:
                                            self.logger.warning("wout.nc 中未找到 DShear/dmier，跳过稳定性惩罚。")
                                else:
                                    self.logger.warning(f"未找到 wout 文件: {self.output_file_path}")
                            except Exception as e:
                                self.logger.warning(f"读取 wout.nc 以获取 Mercier 失败: {e}")
                    
                    analysis_res["min_mercier"] = min_mercier
                    
                    # 允许 0.0 为临界稳定；若 Mercier 数据仍不可用（保持惩罚默认值），不因稳定性单独判为不可行
                    if min_mercier == -999.0:
                        is_stable = True
                    else:
                        is_stable = min_mercier >= 0.0

                except (KeyError, IndexError, AttributeError, ValueError) as e: # 增加了 ValueError
                    self.logger.warning(f"Metric extraction from vmec_output failed: {e}", exc_info=True)
                    self._assign_penalty(item, f"Metric_Extraction_Fail: {e}")
                    invalid_num += 1
                    continue

                # 4. 检查约束
                is_feasible = is_converged and is_stable
                
                raw_results = {
                    'volume': analysis_res.get('volume', 0.0),
                    'aspect_ratio': analysis_res.get('aspect_ratio', 999.0),
                    'magnetic_shear': analysis_res.get('magnetic_shear', 0.0)
                }

                # 5. 应用约束惩罚
                penalized_results = self._apply_penalty(raw_results, is_feasible)
                
                # 6. 转换目标
                transformed = self._transform_objectives(penalized_results)
                overall_score = 1.0 - np.mean(list(transformed.values())) # 框架的 MOO 似乎不需要这个

                results_dict = {
                    'original_results': raw_results,
                    'transformed_results': transformed,
                    'overall_score': overall_score,
                    'constraint_results': {'is_feasible': 1.0 if is_feasible else 0.0, 'is_converged': is_converged, 'is_stable': is_stable, 'min_mercier': min_mercier}
                }
                item.assign_results(results_dict)

            except Exception as e:
                self.logger.critical(f"Unhandled exception during item evaluation: {e}", exc_info=True)
                self._assign_penalty(item, f"Critical_Eval_Error: {e}")
                invalid_num += 1

        return items, { "invalid_num": invalid_num, "repeated_num": 0 }

    def _apply_penalty(self, results: dict, is_feasible: bool) -> dict:
        """如果解不可行，施加惩罚"""
        penalized_results = results.copy()
        if not is_feasible:
            self.logger.warning("Infeasible design (not converged or Mercier unstable). Applying penalty.")
            # 对“越小越好”的目标施加惩罚
            if self.obj_directions.get('aspect_ratio') == 'min':
                penalized_results['aspect_ratio'] *= 10.0 # 惩罚
            # 对“越大越好”的目标施加惩罚
            if self.obj_directions.get('volume') == 'max':
                penalized_results['volume'] *= 0.1 # 惩罚
            if self.obj_directions.get('magnetic_shear') == 'max':
                penalized_results['magnetic_shear'] *= 0.1 # 惩罚 (假设 shear 总是正的)
                
        return penalized_results

    def _assign_penalty(self, item, reason=""):
        """为失败的运行分配一个最差的适应度"""
        penalty_score = 99999
        original = {}
        for obj in self.objs:
            if self.obj_directions[obj] == 'min':
                original[obj] = penalty_score
            else:
                original[obj] = -penalty_score # 或 0

        results = {
            'original_results': original,
            'transformed_results': {obj: 1.0 for obj in self.objs}, # 归一化的最差值
            'overall_score': -1.0,
            'constraint_results': {'is_feasible': 0.0, 'is_converged': False, 'is_stable': False, 'min_mercier': -999.0},
            'error_reason': reason
        }
        item.assign_results(results)

    def _transform_objectives(self, penalized_results: dict) -> dict:
        """
        将原始、带惩罚的目标值转换为 [0, 1] 范围内的归一化分数。
        优化器的目标是最小化这些分数。
        
        **重要提示**: 这些范围 (v_min, v_max 等) 需要根据您对 W7-X 问题的
        先验知识进行调整，以获得最佳的归一化效果。
        """
        transformed = {}
        
        # 1. 体积 (Volume) - 越大越好
        v_min, v_max = 5.0, 50.0 # 示例范围: 5 到 50 m^3
        v = np.clip(penalized_results.get('volume', v_min), v_min, v_max)
        if self.obj_directions.get('volume') == 'max':
            # (*** 已修正 BUG 5 ***)
            transformed['volume'] = (v_max - v) / (v_max - v_min) # 映射: 高 -> 低
        else:
            transformed['volume'] = (v - v_min) / (v_max - v_min) # 映射: 低 -> 低

        # 2. 纵横比 (Aspect Ratio) - 越小越好
        ar_min, ar_max = 5.0, 30.0 # 示例范围: 5 到 30
        ar = np.clip(penalized_results.get('aspect_ratio', ar_max), ar_min, ar_max)
        if self.obj_directions.get('aspect_ratio') == 'min':
            transformed['aspect_ratio'] = (ar - ar_min) / (ar_max - ar_min) # 映射: 低 -> 低
        else:
            transformed['aspect_ratio'] = (ar_max - ar) / (ar_max - ar_min) # 映射: 高 -> 低

        # 3. 磁剪切 (Magnetic Shear) - 越大越好
        s_min, s_max = 0.0, 0.5 # 示例范围: 0.0 到 0.5
        s = np.clip(penalized_results.get('magnetic_shear', s_min), s_min, s_max)
        if self.obj_directions.get('magnetic_shear') == 'max':
            transformed['magnetic_shear'] = (s_max - s) / (s_max - s_min) # 映射: 高 -> 低
        else:
            transformed['magnetic_shear'] = (s - s_min) / (s_max - s_min) # 映射: 低 -> 低
            
        return transformed