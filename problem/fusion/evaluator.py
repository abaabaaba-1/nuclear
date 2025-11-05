# problem/vmecpp/evaluator.py (V3 - 确认版，遵循 SACS 模式)
import numpy as np
import json
import logging
import random
import copy
import re
import vmecpp  # 导入 VMEC++ 库
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from algorithm.base import Item

# --- 1. 助手函数：加载和解析 ---
# (这部分对应 SACS 示例中的 _parse_and_modify_line 助手函数)

def _load_baseline_seed_from_file(filepath: Path) -> Dict:
    """
    (助手函数) 从 input.w7x 文件加载基线种子。
    """
    if not filepath.exists():
        logging.critical(f"VMEC 输入文件未找到: {filepath}")
        raise FileNotFoundError(f"VMEC 输入文件未找到: {filepath}")

    seed_dict = {"new_code_blocks": {}}
    pattern = re.compile(r"^\s*(RBC\(([^)]+)\)\s*=\s*([0-9.eE+-]+)\s*ZBS\(([^)]+)\)\s*=\s*([0-9.eE+-]+))", re.IGNORECASE)

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                full_line = match.group(1).strip()
                rbc_indices_str = match.group(2)
                key = "BC_" + rbc_indices_str.replace(",", "_") 
                seed_dict["new_code_blocks"][key] = full_line
                
    logging.info(f"成功从 {filepath.name} 加载了 {len(seed_dict['new_code_blocks'])} 个边界系数作为基线种子。")
    return seed_dict

def _parse_and_modify_line(line: str, block_name: str) -> str:
    """
    (助手函数) 对 RBC/ZBS "行" 进行诱变。
    这是 GA 脚本 所需的核心突变函数。
    """
    try:
        match = re.search(r"(RBC\([^)]+\)\s*=\s*)([0-9.eE+-]+)(\s*ZBS\([^)]+\)\s*=\s*)([0-9.eE+-]+)", line)
        if not match:
            logging.warning(f"无法解析 VMEC 系数行: {line}")
            return line.rstrip()

        rbc_header, rbc_val_str, zbs_header, zbs_val_str = match.groups()

        # 随机小幅诱变 (例如, +/- 5%)
        rbc_val = float(rbc_val_str) * random.uniform(0.95, 1.05)
        zbs_val = float(zbs_val_str) * random.uniform(0.95, 1.05)

        # 保持原始格式
        new_rbc_str = f"{rbc_val:{len(rbc_val_str)}.4e}"
        new_zbs_str = f"{zbs_val:{len(zbs_val_str)}.4e}"
        
        new_line = f"{rbc_header}{new_rbc_str}{zbs_header}{new_zbs_str}"
        return new_line

    except Exception as e:
        logging.error(f"在 _parse_and_modify_line 中出错 (block: {block_name}): {e}", exc_info=True)
        return line.rstrip()

def _convert_to_str(code_blocks: Dict[str, str]) -> str:
    """
    (助手函数) 将 SACS 风格的 code_blocks 字典转换为 JSON 字符串。
    """
    return json.dumps({"new_code_blocks": code_blocks}, sort_keys=True)


# --- 2. 核心评估器类 (RewardingSystem) ---
# (这对应 SACS 示例中的 RewardingSystem 类)

class RewardingSystem:
    def __init__(self, config):
        """
        初始化 RewardingSystem。
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            base_input_path = config.get('vmec.input_file') 
            if not base_input_path:
                raise ValueError("config.yaml 中缺少 'vmec.input_file' 路径")
                
            self.base_vmec_input = vmecpp.VmecInput.from_file(base_input_path)
            logging.info(f"成功加载基准 VMEC 输入文件: {base_input_path}")
        except FileNotFoundError:
            logging.critical(f"未找到基准 VMEC 输入文件: {base_input_path}")
            raise
            
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

    def evaluate(self, items: List[Item]) -> Tuple[List[Item], Dict]:
        """
        评估 VMEC 候选体列表。
        """
        invalid_num = 0
        num_items = len(items)
        valid_items = []
        
        for item_idx, item in enumerate(items):
            self.logger.info(f"--- [ {item_idx + 1} / {num_items} ] 正在评估 VMEC 候选体... ---")
            
            try:
                # 1. 解析来自 GA/LLM 的 JSON
                raw_value = item.value
                try:
                    if 'candidate' in raw_value:
                        raw_value = raw_value.split('<candidate>', 1)[1].rsplit('</candidate>', 1)[0].strip()
                    modifications = json.loads(raw_value)
                    new_code_blocks = modifications.get("new_code_blocks")
                except (json.JSONDecodeError, IndexError, AttributeError) as e:
                    self.logger.warning(f"无法解析候选 JSON: {raw_value}. 错误: {e}")
                    self._assign_penalty(item, "Invalid JSON format from LLM")
                    invalid_num += 1
                    continue

                if not new_code_blocks or not isinstance(new_code_blocks, dict):
                    self._assign_penalty(item, "Invalid candidate structure (no new_code_blocks)")
                    invalid_num += 1
                    continue
                
                # 2. 构建 VmecInput 对象
                current_input = copy.deepcopy(self.base_vmec_input)
                ntor = current_input.ntor
                
                try:
                    for identifier, line in new_code_blocks.items():
                        match = re.search(r"RBC\(([^)]+)\)\s*=\s*([0-9.eE+-]+)\s*ZBS\(([^)]+)\)\s*=\s*([0-9.eE+-]+)", line)
                        if not match:
                            continue
                        
                        rbc_indices_str, rbc_val_str, zbs_indices_str, zbs_val_str = match.groups()
                        
                        rbc_val = float(rbc_val_str)
                        m_rbc, n_rbc = map(int, rbc_indices_str.split(','))
                        current_input.rbc[m_rbc, ntor + n_rbc] = rbc_val
                        
                        zbs_val = float(zbs_val_str)
                        m_zbs, n_zbs = map(int, zbs_indices_str.split(','))
                        current_input.zbs[m_zbs, ntor + n_zbs] = zbs_val

                except Exception as e:
                    self.logger.warning(f"构建 VmecInput 时出错: {e}")
                    self._assign_penalty(item, f"VMEC_Input_Build_Fail: {e}")
                    invalid_num += 1
                    continue

                # 3. 运行 VMEC++
                try:
                    vmec_output = vmecpp.run(current_input)
                    final_residual = vmec_output.wout.fsqr[-1] 
                    tolerance = vmec_output.wout.ftolv
                    is_converged = final_residual < tolerance
                    
                except RuntimeError as e:
                    self.logger.warning(f"VMEC 分析失败: {e}")
                    self._assign_penalty(item, f"VMEC_Run_Fail: {e}")
                    invalid_num += 1
                    continue
                
                # 4. 提取指标
                try:
                    volume = vmec_output.wout.volume_p  
                    iotas = vmec_output.wout.iotas  
                    magnetic_shear = iotas[-1] - iotas[1] # 边界 - 轴心
                    mercier_values = vmec_output.mercier.Dshear 
                    min_mercier = np.min(mercier_values)
                    
                except Exception as e:
                    self.logger.warning(f"指标提取失败: {e}")
                    self._assign_penalty(item, f"Metric_Extraction_Fail: {e}")
                    invalid_num += 1
                    continue

                # 5. 分配分数
                is_feasible = is_converged and (min_mercier > 0)
                
                raw_results = {
                    'volume': volume,
                    'magnetic_shear': magnetic_shear,
                    'stability': min_mercier
                }

                penalized_results = self._apply_penalty(raw_results, is_feasible, min_mercier, is_converged)
                transformed = self._transform_objectives(penalized_results)
                # MOO 框架假设分数越高越好
                overall_score = 1.0 - np.mean(list(transformed.values())) 

                results_dict = {
                    'original_results': raw_results,
                    'transformed_results': transformed,
                    'overall_score': overall_score,
                    'constraint_results': {
                        'is_feasible': 1.0 if is_feasible else 0.0,
                        'is_converged': 1.0 if is_converged else 0.0,
                        'min_mercier': min_mercier
                    }
                }
                item.assign_results(results_dict)
                valid_items.append(item) # 只有成功的才被添加

            except Exception as e:
                self.logger.critical(f"评估期间发生未处理的异常: {e}", exc_info=True)
                self._assign_penalty(item, f"Critical_Eval_Error: {e}")
                invalid_num += 1
        
        return valid_items, { "invalid_num": invalid_num, "repeated_num": 0 }

    def _apply_penalty(self, results: dict, is_feasible: bool, min_mercier: float, is_converged: bool) -> dict:
        """
        如果解不可行（未收敛或不稳定），则施加惩罚。
        """
        penalized_results = results.copy()
        if not is_feasible:
            penalty_factor = 1.0
            if not is_converged: penalty_factor += 10.0
            if min_mercier <= 0: penalty_factor += 1.0 + abs(min_mercier) * 5.0
            
            self.logger.warning(f"VMEC 不可行: converged={is_converged}, min_mercier={min_mercier:.3e}. 惩罚因子 {penalty_factor:.2f}.")
            
            if self.obj_directions.get('volume') == 'max':
                penalized_results['volume'] /= penalty_factor
            if self.obj_directions.get('magnetic_shear') == 'max':
                penalized_results['magnetic_shear'] /= (penalty_factor * 0.5)
                
        return penalized_results

    def _assign_penalty(self, item, reason=""):
        """
        为灾难性失败分配最差分数。
        """
        penalty_score = -99999
        original = {obj: penalty_score for obj in self.objs}
        results = {
            'original_results': original,
            'transformed_results': {obj: 1.0 for obj in self.objs},
            'overall_score': -1.0, 
            'constraint_results': {'is_feasible': 0.0, 'is_converged': 0.0, 'min_mercier': -999.0},
            'error_reason': reason
        }
        item.assign_results(results)

    def _transform_objectives(self, penalized_results: dict) -> dict:
        """
        将原始惩罚值转换为归一化分数 [0, 1]，其中 0 是 "好"。
        """
        transformed = {}
        
        # --- 1. 体积 (Volume) 变换 (目标: max) ---
        v_min, v_max = 10, 50   # 假设体积范围 (m^3)
        v = np.clip(penalized_results.get('volume', v_min), v_min, v_max)
        transformed['volume'] = (v_max - v) / (v_max - v_min) # 高 v -> 低分 (好)

        # --- 2. 磁剪切 (Magnetic Shear) 变换 (目标: max) ---
        s_min, s_max = -0.5, 1.0  # 假设磁剪切范围
        s = np.clip(penalized_results.get('magnetic_shear', s_min), s_min, s_max)
        transformed['magnetic_shear'] = (s_max - s) / (s_max - s_min) # 高 s -> 低分 (好)
        
        # --- 3. 稳定性 (Stability / Min Mercier) 变换 (目标: max) ---
        m_min, m_max = -0.5, 0.5   # 假设 Mercier 范围
        m = np.clip(penalized_results.get('stability', m_min), m_min, m_max)
        transformed['stability'] = (m_max - m) / (m_max - m_min) # 高 m -> 低分 (好)
            
        return transformed

# --- 3. 核心种群生成器 (generate_initial_population) ---
# (这对应 SACS 示例中的 generate_initial_population)
# (由 MOO.py 调用)

def generate_initial_population(config, seed) -> List[Item]:
    """
    生成初始种群。
    """
    np.random.seed(seed)
    random.seed(seed)
    population_size = config.get('optimization.pop_size')
    
    try:
        base_input_path = Path(config.get('vmec.input_file'))
        SEED_BASELINE = _load_baseline_seed_from_file(base_input_path)
        INITIAL_SEEDS = [SEED_BASELINE]
    except Exception as e:
        logging.critical(f"无法在 generate_initial_population 期间加载基线 VMEC 种子: {e}")
        return [] 

    optimizable_blocks = list(SEED_BASELINE["new_code_blocks"].keys())
    if not optimizable_blocks:
        logging.critical("在种子文件中未找到可优化的块。")
        return []
        
    initial_items = []
    seen_candidates = set()
    max_tries = population_size * 10
    objectives = config.get('goals')

    logging.info(f"正在为 VMEC 生成大小为 {population_size} 的初始种群...")
    
    # 1. 添加基线种子
    for seed_candidate in INITIAL_SEEDS:
        candidate_str = _convert_to_str(seed_candidate["new_code_blocks"])
        if candidate_str not in seen_candidates:
            item = Item(candidate_str, objectives)
            initial_items.append(item)
            seen_candidates.add(candidate_str)
    
    # 2. 生成突变体
    try_count = 0
    while len(initial_items) < population_size and try_count < max_tries:
        base_candidate_dict = copy.deepcopy(random.choice(INITIAL_SEEDS))
        
        num_modifications = random.randint(1, len(optimizable_blocks) // 2)
        blocks_to_modify_names = random.sample(optimizable_blocks, min(num_modifications, len(optimizable_blocks)))

        for block_name in blocks_to_modify_names:
            original_sacs_line = base_candidate_dict["new_code_blocks"][block_name]
            modified_sacs_line = _parse_and_modify_line(original_sacs_line, block_name)
            base_candidate_dict["new_code_blocks"][block_name] = modified_sacs_line
        
        candidate_str = _convert_to_str(base_candidate_dict["new_code_blocks"])
        if candidate_str not in seen_candidates:
            item = Item(candidate_str, objectives)
            initial_items.append(item)
            seen_candidates.add(candidate_str)
        try_count += 1

    if len(initial_items) < population_size:
        logging.warning(f"仅生成了 {len(initial_items)}/{population_size} 个 VMEC 候选体。")

    logging.info(f"成功生成 {len(initial_items)} 个待评估的 VMEC 候选体。")
    return initial_items

# --- 4. (可选) get_database 函数 ---
# (由 MOO.py 调用)
def get_database(config, seed=42, n_sample=100) -> List[Item]:
    """
    模仿 constellaration 的 get_database。
    对于 VMEC++，我们只返回基线种子。
    """
    logging.info("get_database() 被调用... 为 VMEC++ 返回基线种子。")
    try:
        base_input_path = Path(config.get('vmec.input_file'))
        SEED_BASELINE = _load_baseline_seed_from_file(base_input_path)
        objectives = config.get('goals')
        candidate_str = _convert_to_str(SEED_BASELINE["new_code_blocks"])
        item = Item(candidate_str, objectives)
        return [item]
    except Exception as e:
        logging.warning(f"get_database 失败: {e}")
        return []