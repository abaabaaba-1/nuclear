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
from typing import Dict, List, Any, Optional, Tuple

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

COEFF_KEY_PATTERN = re.compile(r"([RZ]B[CS])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)
LOW_ORDER_LIMIT = 0.02
HIGH_ORDER_LIMIT = 0.05
INITIAL_MUTATION_LIMIT = 0.10


def _parse_mode_numbers(key: str) -> Tuple[Optional[int], Optional[int]]:
    norm_key = key.strip().replace(" ", "")
    match = COEFF_KEY_PATTERN.match(norm_key)
    if not match:
        return None, None
    return abs(int(match.group(2))), abs(int(match.group(3)))

# --- 2. 初始种群生成 (来自 SACS 评估器的逻辑) ---

def _mutate_seed_coefficients(seed_coeffs: Dict[str, float]) -> Dict[str, float]:
    """对一组系数进行随机突变以生成新个体"""
    mutated_coeffs = copy.deepcopy(seed_coeffs)
    
    # 随机选择 1 到 5 个系数进行突变
    num_to_mutate = random.randint(1, 5)
    keys_to_mutate = random.sample(list(mutated_coeffs.keys()), num_to_mutate)
    
    for key in keys_to_mutate:
        original_value = mutated_coeffs[key]
        if original_value == 0.0:
            continue

        m, n = _parse_mode_numbers(key)
        rel_limit = HIGH_ORDER_LIMIT
        if m is not None and n is not None and m <= 2 and n <= 1:
            rel_limit = LOW_ORDER_LIMIT
        rel_limit = max(rel_limit, INITIAL_MUTATION_LIMIT)

        mutation_delta = random.uniform(-rel_limit, rel_limit)
        mutated_coeffs[key] = original_value * (1.0 + mutation_delta)
        
    return mutated_coeffs


def _extract_delta_coefficients(base_coeffs: Dict[str, float], mutated_coeffs: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    """仅返回与基准值存在可观差异的系数字典。"""
    deltas = {}
    for key, mutated_val in mutated_coeffs.items():
        base_val = base_coeffs.get(key)
        if base_val is None:
            deltas[key] = mutated_val
        elif abs(mutated_val - base_val) > eps:
            deltas[key] = mutated_val
    return deltas

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
    except Exception as e:
        logging.critical(f"初始化 VmecFileModifier 失败: {e}", exc_info=True)
        raise RuntimeError("无法初始化 VMEC 输入修饰器，请检查 vmec.project_path 和 vmec.input_file 配置。") from e

    if not base_coeffs:
        logging.critical("VmecFileModifier.extract_coefficients() 返回空基准系数映射，请检查 input.w7x 内容。")
        raise RuntimeError("无法从 input.w7x 提取任何 RBC/ZBS 系数，终止初始化。")

    initial_seeds = [{"new_coefficients": {}}]
    
    initial_population_jsons = []
    seen_candidates = set()
    max_tries = population_size * 10

    logging.info(f"正在生成大小为 {population_size} 的初始种群...")
    
    for seed_candidate in initial_seeds:
        candidate_str = json.dumps(seed_candidate)
        if candidate_str not in seen_candidates:
            initial_population_jsons.append(candidate_str)
            seen_candidates.add(candidate_str)
    
    try_count = 0
    while len(initial_population_jsons) < population_size and try_count < max_tries:
        # 应用突变
        mutated_coeffs = _mutate_seed_coefficients(base_coeffs)
        delta_coeffs = _extract_delta_coefficients(base_coeffs, mutated_coeffs)

        if not delta_coeffs:
            try_count += 1
            continue

        max_changes = config.get('llm_constraints.max_coeff_changes', 12)
        if max_changes and len(delta_coeffs) > max_changes:
            delta_keys = list(delta_coeffs.keys())
            random.shuffle(delta_keys)
            delta_keys = delta_keys[:max_changes]
            delta_coeffs = {key: delta_coeffs[key] for key in delta_keys}

        candidate_payload = {"new_coefficients": delta_coeffs}
        candidate_str = json.dumps(candidate_payload)
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
        
        # wout文件保存控制
        self.output_file_path = Path(self.sacs_project_path) / config.get('vmec.output_file')
        self.save_wout_mode = config.get('vmec.save_wout_mode', 'none')  # none | debug | top_k | all
        self.save_wout_top_k = config.get('vmec.save_wout_top_k', 10)
        self.wout_save_counter = 0  # 用于跟踪保存的wout文件数量
        
        if self.save_wout_mode not in ['none', 'debug', 'top_k', 'all']:
            self.logger.warning(f"Invalid save_wout_mode '{self.save_wout_mode}', defaulting to 'none'")
            self.save_wout_mode = 'none'
        
        self.logger.info(f"WOUT save mode: {self.save_wout_mode}" + 
                        (f" (top {self.save_wout_top_k})" if self.save_wout_mode == 'top_k' else ""))
        
        self.modifier = VmecFileModifier(self.sacs_project_path, self.input_file)
        try:
            extracted = self.modifier.extract_coefficients()
        except Exception as base_e:
            self.logger.critical(
                f"Failed to extract baseline coefficients from VMEC input: {base_e}",
                exc_info=True,
            )
            raise RuntimeError(
                "Cannot initialize RewardingSystem: failed to extract baseline VMEC coefficients."
            ) from base_e

        if not extracted:
            self.logger.critical(
                "VmecFileModifier.extract_coefficients() returned an empty coefficient map. "
                "Please verify 'vmec.project_path' and 'vmec.input_file' in the config."
            )
            raise RuntimeError("Cannot initialize RewardingSystem with empty baseline coefficient map.")

        self.base_coeffs = extracted
        
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}
        self.obj_ranges = config.get('objective_ranges', {})
        self.llm_constraints = {
            'max_coeff_changes': config.get('llm_constraints.max_coeff_changes', 8),
            'low_order_max_rel_change': config.get('llm_constraints.low_order_max_rel_change', 0.02),
            'high_order_max_rel_change': config.get('llm_constraints.high_order_max_rel_change', 0.05),
        }
        
        # Top-K候选跟踪器（用于save_wout_mode='top_k'）
        self.top_k_candidates = []  # 存储 (total_score, candidate_id, is_saved) 元组

    def evaluate(self, items):
        invalid_num = 0
        for item in items:
            analysis_res = {} # 存储从 vmec_output 提取的指标
            is_feasible = False
            is_converged = False
            is_stable = False
            min_mercier: Optional[float] = None

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
                
                # 1. 修改 input.w7x 文件（在基准系数上叠加增量，防止遗漏键）
                sanitized_coeffs = self._sanitize_new_coefficients(new_coefficients)
                merged_coeffs = copy.deepcopy(self.base_coeffs)
                merged_coeffs.update(sanitized_coeffs)
                if not self.modifier.replace_coefficients(merged_coeffs):
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
                    
                    # 智能保存wout文件（根据save_wout_mode设置）
                    # 注意：此时还未计算total score，稍后在top_k模式中会重新保存
                    should_save_now = (self.save_wout_mode == 'all' or 
                                      (self.save_wout_mode == 'debug' and self.wout_save_counter < 5))
                    
                    if should_save_now:
                        try:
                            vmec_output.wout.save(self.output_file_path)
                            self.wout_save_counter += 1
                            self.logger.debug(f"WOUT saved (mode={self.save_wout_mode}, count={self.wout_save_counter})")
                        except Exception as save_e:
                            self.logger.warning(f"Failed to save wout.nc: {save_e}")
                    
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
                    
                    # 添加收敛性调试日志
                    if not is_converged:
                        self.logger.debug(f"Convergence check failed: fsqr={fsqr:.2e} (tol={tolerance:.2e}), "
                                        f"fsqz={fsqz:.2e}, fsql={fsql:.2e}")
                    
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
                    if len(iotas_profile) >= 2:
                        iota_at_axis = iotas_profile[0]
                        iota_at_edge = iotas_profile[-1]
                    else:
                        iota_at_axis = iotas_profile[0]
                        iota_at_edge = iotas_profile[0]
                    magnetic_shear = iota_at_edge - iota_at_axis
                    analysis_res["iota_axis"] = iota_at_axis
                    analysis_res["iota_edge"] = iota_at_edge
                    analysis_res["magnetic_shear"] = magnetic_shear # 目标

                    # 3d. 纵横比 (目标)
                    aspect_ratio = None
                    if hasattr(vmec_output.wout, "Rmajor_p") and hasattr(vmec_output.wout, "Aminor_p"):
                        try:
                            R_major_val = float(vmec_output.wout.Rmajor_p)
                            a_minor_val = float(vmec_output.wout.Aminor_p)
                            if a_minor_val != 0.0:
                                aspect_ratio = R_major_val / a_minor_val
                        except Exception:
                            aspect_ratio = None
                    if aspect_ratio is None:
                        rmnc = vmec_output.wout.rmnc # 期望形状 (mnmax, ns)
                        xm = vmec_output.wout.xm     # 期望形状 (mnmax,)
                        ns = vmec_output.wout.ns     # 标量，磁面数量
                        
                        # 维度安全检查
                        if not hasattr(rmnc, 'shape') or len(rmnc.shape) != 2:
                            self.logger.error(f"Invalid rmnc shape: expected 2D array, got {getattr(rmnc, 'shape', 'no shape')}")
                            aspect_ratio = 999.0  # 使用惩罚值
                        elif rmnc.shape[1] < 2:
                            self.logger.error(f"Insufficient radial surfaces: rmnc has only {rmnc.shape[1]} surfaces")
                            aspect_ratio = 999.0
                        else:
                            last_surface_idx = rmnc.shape[1] - 1  # 使用实际的最后一列索引
                            
                            # 验证xm维度匹配
                            if xm.shape[0] != rmnc.shape[0]:
                                self.logger.warning(f"Dimension mismatch: xm.shape={xm.shape}, rmnc.shape={rmnc.shape}")
                            
                            # 计算最外层磁面的主半径和小半径
                            # R_outer: theta=0处的R值, R_inner: theta=pi处的R值
                            R_outer = np.dot(rmnc[:, last_surface_idx], np.cos(xm * 0.0))
                            R_inner = np.dot(rmnc[:, last_surface_idx], np.cos(xm * np.pi))
                            
                            R_major = (R_outer + R_inner) / 2.0
                            a_minor = (R_outer - R_inner) / 2.0
                            
                            if a_minor > 0:
                                aspect_ratio = R_major / a_minor
                            else:
                                self.logger.error(f"Invalid minor radius: a_minor={a_minor}")
                                aspect_ratio = 999.0
                            
                            self.logger.debug(f"Aspect ratio calculation: R_outer={R_outer:.4f}, R_inner={R_inner:.4f}, "
                                            f"R_major={R_major:.4f}, a_minor={a_minor:.4f}, AR={aspect_ratio:.4f}")
                    analysis_res["aspect_ratio"] = aspect_ratio # 目标

                    # 3e. Mercier 稳定性 (约束)
                    min_mercier = None
                    mercier_source = "unknown"
                    
                    # 尝试多个来源获取Mercier数据（优先级递减）
                    mercier_candidates = [
                        (lambda: vmec_output.mercier.DShear if hasattr(vmec_output, 'mercier') and vmec_output.mercier and hasattr(vmec_output.mercier, 'DShear') else None, "mercier.DShear"),
                        (lambda: vmec_output.mercier.dmerc if hasattr(vmec_output, 'mercier') and vmec_output.mercier and hasattr(vmec_output.mercier, 'dmerc') else None, "mercier.dmerc"),
                        (lambda: vmec_output.wout.DShear if hasattr(vmec_output.wout, 'DShear') else None, "wout.DShear"),
                        (lambda: vmec_output.wout.dmerc if hasattr(vmec_output.wout, 'dmerc') else None, "wout.dmerc"),
                    ]
                    
                    for getter, source_name in mercier_candidates:
                        try:
                            data = getter()
                            if data is not None and hasattr(data, 'shape') and len(data) > 0:
                                min_mercier = float(np.min(data))
                                mercier_source = source_name
                                break
                        except Exception as e:
                            self.logger.debug(f"Failed to get Mercier from {source_name}: {e}")
                            continue
                    
                    # 如果内存中获取失败，尝试从磁盘读取（最后手段）
                    if min_mercier is None and self.output_file_path.is_file():
                        try:
                            from netCDF4 import Dataset
                            with Dataset(str(self.output_file_path), 'r') as ds:
                                ds.set_always_mask(False)
                                for var_name in ['DShear', 'dmerc', 'dmier']:
                                    if var_name in ds.variables:
                                        min_mercier = float(np.min(ds.variables[var_name][()]))
                                        mercier_source = f"disk.{var_name}"
                                        break
                        except Exception as e:
                            self.logger.debug(f"Failed to read Mercier from disk: {e}")
                    
                    # 设置默认值和稳定性判定
                    if min_mercier is None:
                        min_mercier = -999.0  # 惩罚值
                        is_stable = False
                        self.logger.warning("Mercier data unavailable from all sources; marking design as unstable.")
                    else:
                        is_stable = min_mercier >= 0.0
                        self.logger.debug(f"Mercier criterion: min={min_mercier:.4f} (source: {mercier_source}), stable={is_stable}")
                    
                    analysis_res["min_mercier"] = min_mercier

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

        # 关键：必须恢复baseline系数，否则后续评估会出错
        try:
            if not self.modifier.replace_coefficients(self.base_coeffs):
                raise RuntimeError("Failed to restore baseline coefficients (replace_coefficients returned False)")
        except Exception as restore_e:
            self.logger.critical(f"CRITICAL: Failed to restore baseline VMEC input after evaluation: {restore_e}")
            self.logger.critical("This will cause all subsequent evaluations to use incorrect baseline!")
            # 尝试紧急恢复：从最近的备份恢复
            backup_files = sorted(self.modifier.backup_dir.glob(f"{self.modifier.input_file_path.name}.backup_*"))
            if backup_files:
                try:
                    latest_backup = backup_files[-1]
                    self.modifier._restore_from_backup(latest_backup)
                    self.logger.warning(f"Emergency restore from backup: {latest_backup.name}")
                except Exception as e2:
                    self.logger.critical(f"Emergency restore also failed: {e2}")
                    raise RuntimeError("Cannot continue: baseline VMEC input is corrupted") from restore_e
            else:
                raise RuntimeError("Cannot continue: no backup available to restore baseline") from restore_e

        return items, { "invalid_num": invalid_num, "repeated_num": 0 }

    def _sanitize_new_coefficients(self, new_coefficients: dict) -> dict:
        max_changes = self.llm_constraints.get('max_coeff_changes', 8)
        low_limit = self.llm_constraints.get('low_order_max_rel_change', 0.02)
        high_limit = self.llm_constraints.get('high_order_max_rel_change', 0.05)
        keys = list(new_coefficients.keys())
        if max_changes is not None and max_changes > 0 and len(keys) > max_changes:
            keys = keys[:max_changes]
        sanitized = {}
        for key in keys:
            value = new_coefficients[key]
            norm_key = key.strip().replace(" ", "")
            base_val = self.base_coeffs.get(norm_key)
            m, n = _parse_mode_numbers(norm_key)
            if base_val is not None and base_val != 0.0 and m is not None and n is not None:
                limit = low_limit if (m <= 2 and n <= 1) else high_limit
                rel_change = abs(value - base_val) / abs(base_val)
                if rel_change > limit:
                    if value >= base_val:
                        value = base_val * (1.0 + limit)
                    else:
                        value = base_val * (1.0 - limit)
            sanitized[norm_key] = value
        return sanitized

    def _apply_penalty(self, results: dict, is_feasible: bool) -> dict:
        """如果解不可行，施加惩罚"""
        penalized_results = results.copy()
        if is_feasible:
            return penalized_results

        self.logger.warning("Infeasible design (not converged or Mercier unstable). Applying penalty.")
        for obj in self.objs:
            direction = self.obj_directions.get(obj)
            obj_range = self.obj_ranges.get(obj, None)
            if direction == 'min':
                if obj_range and len(obj_range) == 2:
                    penalized_results[obj] = obj_range[1]
                else:
                    penalized_results[obj] = results.get(obj, 0.0) * 10.0
            elif direction == 'max':
                if obj_range and len(obj_range) == 2:
                    penalized_results[obj] = obj_range[0]
                else:
                    penalized_results[obj] = results.get(obj, 0.0) * 0.1

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
        v_range = self.obj_ranges.get('volume', [5.0, 50.0])
        v_min, v_max = v_range[0], v_range[1]
        v = np.clip(penalized_results.get('volume', v_min), v_min, v_max)
        if self.obj_directions.get('volume') == 'max':
            # (*** 已修正 BUG 5 ***)
            transformed['volume'] = (v_max - v) / (v_max - v_min) # 映射: 高 -> 低
        else:
            transformed['volume'] = (v - v_min) / (v_max - v_min) # 映射: 低 -> 低

        # 2. 纵横比 (Aspect Ratio) - 越小越好
        ar_range = self.obj_ranges.get('aspect_ratio', [5.0, 30.0])
        ar_min, ar_max = ar_range[0], ar_range[1]
        ar = np.clip(penalized_results.get('aspect_ratio', ar_max), ar_min, ar_max)
        if self.obj_directions.get('aspect_ratio') == 'min':
            transformed['aspect_ratio'] = (ar - ar_min) / (ar_max - ar_min) # 映射: 低 -> 低
        else:
            transformed['aspect_ratio'] = (ar_max - ar) / (ar_max - ar_min) # 映射: 高 -> 低

        # 3. 磁剪切 (Magnetic Shear) - 越大越好
        s_range = self.obj_ranges.get('magnetic_shear', [0.0, 0.5])
        s_min, s_max = s_range[0], s_range[1]
        s = np.clip(penalized_results.get('magnetic_shear', s_min), s_min, s_max)
        if self.obj_directions.get('magnetic_shear') == 'max':
            transformed['magnetic_shear'] = (s_max - s) / (s_max - s_min) # 映射: 高 -> 低
        else:
            transformed['magnetic_shear'] = (s - s_min) / (s_max - s_min) # 映射: 低 -> 低
            
        return transformed
