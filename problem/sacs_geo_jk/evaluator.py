# evaluator.py (已更新 - 采用“小幅度概念种群”策略)
import numpy as np
import json
import logging
import random
import copy
import re
from pathlib import Path

# --- 【确认】使用相对路径导入 ---
from .sacs_file_modifier import SacsFileModifier
from .sacs_runner import SacsRunner
from .sacs_interface_uc import get_sacs_uc_summary
from .sacs_interface_weight_improved import calculate_sacs_weight_from_db
# --- 导入结束 ---

# -------------------------------------------------------------------------
# [保留] 您原有的健壮的SACS行处理函数 (无需修改)
# -------------------------------------------------------------------------

def _parse_and_modify_line(original_line: str, block_name: str, config=None) -> str:
    """
    Parses and modifies a JOINT line using a robust, surgical replacement strategy
    that preserves the original file's exact formatting, including spacing and precision.
    (此函数来自您提供的 evaluator.py)
    """
    try:
        keyword = block_name.split()[0]
        if keyword != "JOINT":
            return original_line.rstrip()

        # 1. Find all numbers and their precise start/end positions in the string.
        num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
        matches = list(num_pattern.finditer(original_line))

        if len(matches) < 3:
            logging.warning(f"Could not find at least 3 coordinates in JOINT line: {original_line.rstrip()}")
            return original_line.rstrip()

        # 2. Get random mutation parameters
        amplitude_range = 2.0
        if config:
            amplitudes = config.get('optimization.mutation_strategy.joint_mutation_amplitudes', {})
            if amplitudes:
                chosen_amplitude_key = random.choice(list(amplitudes.keys()))
                amplitude_range = amplitudes.get(chosen_amplitude_key, 2.0)
        
        max_steps = int(amplitude_range / 0.01)
        num_steps = random.randint(-max_steps, max_steps) if max_steps > 0 else 0
        if num_steps == 0: num_steps = 1 # Ensure some change
        change = num_steps * 0.01

        # 3. Select which coordinate (X, Y, or Z) to modify
        coord_indices = {'x': 0, 'y': 1, 'z': 2}
        coord_to_change_name = random.choice(['x', 'y', 'z'])
        target_match_index = coord_indices[coord_to_change_name]
        target_match = matches[target_match_index]

        # 4. Perform the mutation
        original_value = float(target_match.group(0))
        new_value = original_value + change

        # 5. Dynamically format the new value to perfectly match the old value's format
        original_text = target_match.group(0)
        original_len = len(original_text)
        
        if '.' in original_text:
            precision = len(original_text.split('.')[1])
        else:
            precision = 0
        
        format_spec = f">{original_len}.{precision}f"
        new_text = format(new_value, format_spec)

        if len(new_text) > original_len:
            new_text = new_text[:original_len]

        # 6. Surgically replace the old number text with the new formatted text
        start, end = target_match.span()
        new_line = original_line[:start] + new_text + original_line[end:]
        
        return new_line.rstrip()

    except Exception as e:
        logging.error(f"CRITICAL error in _parse_and_modify_line for '{block_name}': {e}", exc_info=True)
        return original_line.rstrip()

def _get_coords_from_modified_line(modified_line: str) -> dict:
    """A simple parser to get the first three float values from a line."""
    try:
        num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
        all_numbers = [float(n) for n in num_pattern.findall(modified_line)]
        if len(all_numbers) < 3: return None
        return {'x': all_numbers[0], 'y': all_numbers[1], 'z': all_numbers[2]}
    except Exception:
        return None

def _build_slave_joint_line(original_slave_line: str, master_coords: dict) -> str:
    """
    Rebuilds a slave joint line using the coordinates from its master.
    It uses the same robust, format-preserving logic as _parse_and_modify_line.
    (此函数来自您提供的 evaluator.py)
    """
    try:
        num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
        matches = list(num_pattern.finditer(original_slave_line))
        if len(matches) < 3: return original_slave_line.rstrip()

        x_match, y_match, z_match = matches[0], matches[1], matches[2]
        
        replacements = [
            {'match': x_match, 'value': master_coords['x']},
            {'match': y_match, 'value': master_coords['y']},
            {'match': z_match, 'value': master_coords['z']}
        ]
        
        line_editor = list(original_slave_line)
        for rep in reversed(replacements):
            match = rep['match']
            value = rep['value']
            
            original_text = match.group(0)
            original_len = len(original_text)
            precision = len(original_text.split('.')[1]) if '.' in original_text else 0
            
            format_spec = f">{original_len}.{precision}f"
            new_text = format(value, format_spec)
            if len(new_text) > original_len: new_text = new_text[:original_len]
            
            start, end = match.span()
            line_editor[start:end] = list(new_text)
            
        return "".join(line_editor).rstrip()

    except Exception as e:
        logging.error(f"CRITICAL error in _build_slave_joint_line: {e}", exc_info=True)
        return original_slave_line.rstrip()

def _get_initial_joint_definitions(config: dict) -> dict:
    """ (此函数来自您提供的 evaluator.py) """
    sacs_file_path = config.get('sacs.project_path')
    if not sacs_file_path:
        logging.error("Config missing 'sacs.project_path'.")
        return {}

    sacs_file = Path(sacs_file_path) / "sacinp.demo06"
    if not sacs_file.exists():
        sacs_file = Path(sacs_file_path) / "sacinp.txt" # Fallback

    optimizable_joints_list = config.get('sacs.optimizable_joints', [])
    coupled_joints_map = config.get('sacs.coupled_joints', {})
    slave_joints_list = [f"JOINT {v}" for v in coupled_joints_map.values()]
    joint_lines, all_target_joints = {}, set(optimizable_joints_list + slave_joints_list)

    try:
        with open(sacs_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for joint_prefix in all_target_joints:
            parts = joint_prefix.split()
            if len(parts) != 2: continue
            keyword, id_val = parts
            pattern = re.compile(r"^\s*" + re.escape(keyword) + r"\s+" + re.escape(id_val))
            for line in lines:
                if pattern.search(line):
                    key = joint_prefix.replace(" ", "_")
                    joint_lines[key] = line.rstrip('\n')
                    break
    except FileNotFoundError:
        logging.error(f"SACS input file not found at {sacs_file} for seed generation.")

    return joint_lines

# -------------------------------------------------------------------------
# [保留] 您的基线 SEED (无需修改)
# -------------------------------------------------------------------------
SEED_BASELINE = {"new_code_blocks": {"GRUP_LG1":"GRUP LG1         42.200 1.450 29.0011.6050.00 1    1.001.00     0.500N490.005.00","GRUP_LG2":"GRUP LG2         42.200 1.450 29.0011.6050.00 1    1.001.00     0.500N490.006.15","GRUP_LG3":"GRUP LG3         42.200 1.450 29.0011.6050.00 1    1.001.00     0.500N490.006.75","GRUP_LG4":"GRUP LG4         42.200 1.450 29.0011.6050.00 1    1.001.00     0.500N490.00","GRUP_LG5":"GRUP LG5         36.300 1.050 29.0011.6050.00 1    1.001.00     0.500N490.00","GRUP_LG6":"GRUP LG6         36.300 0.800 29.0011.0036.00 1    1.001.00     0.500N490.003.25","GRUP_LG7":"GRUP LG7         26.200 0.800 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_PL1":"GRUP PL1         36.300 1.050 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_PL2":"GRUP PL2         36.300 1.050 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_PL3":"GRUP PL3         36.300 1.050 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_PL4":"GRUP PL4         36.300 1.050 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_T01":"GRUP T01         16.100 0.650 29.0111.2035.00 1    1.001.00     0.500N490.00","GRUP_T02":"GRUP T02         20.100 0.780 29.0011.6035.00 1    1.001.00     0.500N490.00","GRUP_T03":"GRUP T03         12.800 0.520 29.0111.6035.00 1    1.001.00     0.500N490.00","GRUP_T04":"GRUP T04         24.100 0.780 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_T05":"GRUP T05         26.100 1.050 29.0011.6036.00 1    1.001.00     0.500N490.00","GRUP_W.B":"GRUP W.B         36.500 1.050 29.0111.2035.97 1    1.001.00     0.500 490.00","GRUP_W01":"GRUP W01 W24X162              29.0111.2035.97 1    1.001.00     0.500 490.00","GRUP_W02":"GRUP W02 W24X131              29.0111.2035.97 1    1.001.00     0.500 490.00","PGRUP_P01":"PGRUP P01 0.3750I29.000 0.25036.000                                     490.0000"}}

# -------------------------------------------------------------------------
# [新增] 策略3：原始的随机抖动 (封装了您之前的逻辑)
# -------------------------------------------------------------------------
def _create_random_mutation_candidate(base_seed, joint_info):
    """
    策略3：原始的随机抖动方法，以保持种群多样性。
    (此逻辑来自您 evaluator.py 的 generate_initial_population)
    """
    candidate = copy.deepcopy(base_seed)
    config = joint_info['config']
    optimizable_joints = joint_info['optimizable_joints']
    coupled_joints_map = joint_info['coupled_joints_map']

    if not optimizable_joints:
        return candidate # 没有可优化的，返回原样

    # 随机决定修改多少个节点
    num_modifications = random.randint(1, max(1, len(optimizable_joints) // 2))
    items_to_modify = random.sample(optimizable_joints, min(num_modifications, len(optimizable_joints)))

    for item_name in items_to_modify:
        item_key = item_name.replace(" ", "_")
        if item_key not in candidate["new_code_blocks"]: continue
        
        original_sacs_line = candidate["new_code_blocks"][item_key]
        # 使用您原有的、健壮的行修改器
        modified_sacs_line = _parse_and_modify_line(original_sacs_line, item_name, config=config)
        candidate["new_code_blocks"][item_key] = modified_sacs_line

        # 必须处理耦合节点
        if item_name.startswith("JOINT"):
            joint_id = item_name.split(" ")[1]
            if joint_id in coupled_joints_map:
                slave_id = coupled_joints_map[joint_id]
                slave_key = f"JOINT_{slave_id}"
                # 获取新坐标并更新从节点
                master_coords = _get_coords_from_modified_line(modified_sacs_line)
                if master_coords and slave_key in candidate["new_code_blocks"]:
                    original_slave_line = candidate["new_code_blocks"][slave_key]
                    new_slave_line = _build_slave_joint_line(original_slave_line, master_coords)
                    candidate["new_code_blocks"][slave_key] = new_slave_line

    return candidate

# -------------------------------------------------------------------------
# [新增] “概念性”修改的辅助函数
# -------------------------------------------------------------------------
def _apply_conceptual_changes(base_seed, joint_info, modifications):
    """
    辅助函数：将“概念性”修改（一个包含多项坐标变化的字典）应用到候选者上。
    modifications 示例: {'JOINT 201': {'x': 1.5, 'y': 1.5}, 'JOINT 203': {'x': 1.5, 'y': 1.5}}
    """
    try:
        candidate = copy.deepcopy(base_seed)
        coupled_joints_map = joint_info['coupled_joints_map']
        
        # 跟踪已修改的主节点，以便后续更新其从节点
        modified_master_joints = {} # 格式: { 'JOINT 201': '...new_line_content...' }

        for joint_name, coords_to_change in modifications.items():
            item_key = joint_name.replace(" ", "_")
            if item_key not in candidate["new_code_blocks"]:
                logging.warning(f"Joint key {item_key} not in seed. Skipping.")
                continue
                
            original_line = candidate["new_code_blocks"][item_key]
            
            # --- 复用 _parse_and_modify_line 的高精度格式化逻辑 ---
            num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
            matches = list(num_pattern.finditer(original_line))
            if len(matches) < 3: continue

            coord_indices = {'x': 0, 'y': 1, 'z': 2}
            new_line = list(original_line) # 可编辑的行内容

            # 从右到左应用更改 (z, y, x)，以防索引错位
            for coord_name in sorted(coords_to_change.keys(), reverse=True):
                if coord_name not in coord_indices: continue
                
                change_val = coords_to_change[coord_name]
                target_match = matches[coord_indices[coord_name]]
                
                original_value = float(target_match.group(0))
                new_value = original_value + change_val
                
                # 动态格式化，保持SACS文件格式不变
                original_text = target_match.group(0)
                original_len = len(original_text)
                precision = len(original_text.split('.')[1]) if '.' in original_text else 0
                format_spec = f">{original_len}.{precision}f"
                new_text = format(new_value, format_spec)
                if len(new_text) > original_len: new_text = new_text[:original_len]
                
                start, end = target_match.span()
                new_line[start:end] = list(new_text)
            
            modified_sacs_line = "".join(new_line).rstrip()
            candidate["new_code_blocks"][item_key] = modified_sacs_line
            # --- 格式化逻辑结束 ---
            
            # 如果这个节点是主节点，记录下它的新行
            joint_id = joint_name.split(" ")[1]
            if joint_id in coupled_joints_map:
                modified_master_joints[joint_name] = modified_sacs_line
        
        # --- 独立循环：处理所有耦合节点 ---
        # 确保所有主节点都改完后，再统一修改从节点
        for joint_name, modified_sacs_line in modified_master_joints.items():
            joint_id = joint_name.split(" ")[1]
            slave_id = coupled_joints_map[joint_id]
            slave_key = f"JOINT_{slave_id}"
            
            master_coords = _get_coords_from_modified_line(modified_sacs_line)
            if master_coords and slave_key in candidate["new_code_blocks"]:
                original_slave_line = candidate["new_code_blocks"][slave_key]
                new_slave_line = _build_slave_joint_line(original_slave_line, master_coords)
                candidate["new_code_blocks"][slave_key] = new_slave_line
            else:
                 logging.warning(f"Could not update slave joint {slave_key}")

        return candidate
    except Exception as e:
        logging.error(f"CRITICAL error in _apply_conceptual_changes: {e}", exc_info=True)
        return None

# -------------------------------------------------------------------------
# [新增] 策略1：V型/A型轮廓 (小幅度，SACS安全)
# -------------------------------------------------------------------------
def _create_v_shape_candidate(base_seed, joint_info):
    """
    策略1：生成一个V型（或A型）轮廓的候选者。
    使底部节点（如600层）更宽/窄，顶部节点（如200层）相反。
    """
    modifications = {}
    optimizable_joints = joint_info['optimizable_joints']
    
    # [关键]：使用SACS安全的小幅度范围 (0.5 到 2.5 米)
    amplitude = random.uniform(0.5, 2.5) * random.choice([-1, 1]) 
    
    for joint_name in optimizable_joints:
        if not joint_name.startswith("JOINT"): continue
        
        try:
            joint_id_str = joint_name.split(" ")[1]
            if not joint_id_str.isdigit(): continue
            joint_level = int(joint_id_str[0]) # 2, 3, 4, 5, 6...
        except (IndexError, ValueError):
            continue # 忽略格式不正确的JOINT
            
        # 变化幅度与层高成比例:
        # 假设 200/300 是顶部, 500/600 是底部
        if joint_level in [2, 3]:
            scale = -0.5 # 顶部反向 (收缩/扩张)
        elif joint_level in [5, 6]:
            scale = 1.0  # 底部完全变化 (扩张/收缩)
        else: # 400层
            scale = 0.25 # 中间层轻微变化
            
        # 仅当节点坐标不是0时才应用（避免移动中心线节点）
        # 这是一个假设，可能需要根据您的SACS模型调整
        # 为简单起见，我们先假设所有可优化节点都在X,Y平面上
        
        change = amplitude * scale
        if abs(change) < 0.01: continue

        # 同时改变 X 和 Y 坐标以实现“扩张”或“收缩”
        # 注意：这里假设了 'x' 和 'y' 的变化是相同的
        # 如果您的模型是矩形的，可能需要分别处理
        modifications[joint_name] = {'x': change, 'y': change}
        
    if not modifications:
        return _create_random_mutation_candidate(base_seed, joint_info) # 如果没生成变化，则回退

    return _apply_conceptual_changes(base_seed, joint_info, modifications)

# -------------------------------------------------------------------------
# [新增] 策略2：阶梯型 (小幅度，SACS安全)
# -------------------------------------------------------------------------
def _create_tiered_setback_candidate(base_seed, joint_info):
    """
    策略2：生成一个“阶梯型”候选者。
    随机选择一个中间层（如400或500层），并使其显著变宽或变窄。
    """
    modifications = {}
    optimizable_joints = joint_info['optimizable_joints']
    
    target_level = random.choice([3, 4, 5]) # 随机选 300, 400 或 500 层
    # [关键]：使用SACS安全的小幅度范围 (0.5 到 2.5 米)
    amplitude = random.uniform(0.5, 2.5) * random.choice([-1, 1]) 
    
    for joint_name in optimizable_joints:
        if not joint_name.startswith("JOINT"): continue
        
        try:
            joint_id_str = joint_name.split(" ")[1]
            if not joint_id_str.isdigit(): continue
            joint_level = int(joint_id_str[0])
        except (IndexError, ValueError):
            continue
        
        if joint_level == target_level:
            modifications[joint_name] = {'x': amplitude, 'y': amplitude}

    if not modifications:
        return _create_random_mutation_candidate(base_seed, joint_info) # 回退

    return _apply_conceptual_changes(base_seed, joint_info, modifications)


# -------------------------------------------------------------------------
# [新增] 策略4：径向缩放（分层等比例缩放X/Y，保持对称性）
# -------------------------------------------------------------------------
def _create_radial_scale_candidate(base_seed, joint_info):
    """
    策略4：对各层进行小幅度的等比例径向缩放（X/Y同向同幅），
    既能探索截面开合，又较稳定地保持整体对称性与可行性。
    """
    modifications = {}
    optimizable_joints = joint_info['optimizable_joints']

    # 每层一个缩放因子，范围在 [-2.0, 2.0] 米的等效位移（而非乘法）
    # 选择较小幅度，避免早期大量不可行
    level_to_delta = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    for lvl in level_to_delta.keys():
        level_to_delta[lvl] = random.uniform(-1.5, 1.5)

    for joint_name in optimizable_joints:
        if not joint_name.startswith("JOINT"):
            continue
        try:
            joint_id_str = joint_name.split(" ")[1]
            if not joint_id_str.isdigit():
                continue
            joint_level = int(joint_id_str[0])
        except (IndexError, ValueError):
            continue

        if joint_level in level_to_delta:
            delta = level_to_delta[joint_level]
            if abs(delta) < 0.01:
                continue
            modifications[joint_name] = {'x': delta, 'y': delta}

    if not modifications:
        return _create_random_mutation_candidate(base_seed, joint_info)

    return _apply_conceptual_changes(base_seed, joint_info, modifications)

# -------------------------------------------------------------------------
# [核心替换] 新的初始种群生成函数
# -------------------------------------------------------------------------
def generate_initial_population(config, seed):
    np.random.seed(seed); random.seed(seed)
    population_size = config.get('optimization.pop_size')
    optimizable_joints_list = config.get('sacs.optimizable_joints', [])
    coupled_joints_map = config.get('sacs.coupled_joints', {})
    initial_joint_lines = _get_initial_joint_definitions(config)

    if not initial_joint_lines:
        logging.critical("FATAL: Could not load any JOINT definitions from the SACS input file.");
        return [json.dumps(SEED_BASELINE, sort_keys=True)]

    logging.info(f"Successfully loaded {len(initial_joint_lines)} initial JOINT definitions from SACS file.")
    master_seed = copy.deepcopy(SEED_BASELINE)
    master_seed["new_code_blocks"].update(initial_joint_lines)
    
    initial_population_jsons, seen_candidates = [], set()
    master_seed_str = json.dumps(master_seed, sort_keys=True)
    initial_population_jsons.append(master_seed_str) # 添加未修改的基线
    seen_candidates.add(master_seed_str)
    
    logging.info(f"Starting generation of CONCEPTUALLY DIVERSE initial population of size {population_size}...")
    max_tries, try_count = population_size * 20, 0 # 增加尝试次数以防重复

    # --- 新的“概念”策略列表 ---
    strategies = [
        _create_v_shape_candidate,        # 策略1：V型/A型 (占1/3)
        _create_tiered_setback_candidate, # 策略2：阶梯型 (占1/3)
        _create_radial_scale_candidate,   # 策略4：径向缩放 (新增)
        _create_random_mutation_candidate # 策略3：原始的随机抖动
    ]
    
    # 将所需信息打包，传递给策略函数
    joint_info = {
        'optimizable_joints': optimizable_joints_list,
        'coupled_joints_map': coupled_joints_map,
        'config': config # 随机抖动策略需要config
    }
    # ---------------------------

    while len(initial_population_jsons) < population_size and try_count < max_tries:
        
        # 随机选择一个“概念”策略
        strategy_func = random.choice(strategies)
        
        # 传入 master_seed 和 joint_info 来生成候选
        base_candidate = strategy_func(master_seed, joint_info)
        
        if not base_candidate:
            try_count += 1
            logging.warning("Strategy function returned None. Retrying.")
            continue

        candidate_str = json.dumps(base_candidate, sort_keys=True)
        if candidate_str not in seen_candidates:
            initial_population_jsons.append(candidate_str)
            seen_candidates.add(candidate_str)
        try_count += 1

    if len(initial_population_jsons) < population_size:
        logging.warning(f"Only generated {len(initial_population_jsons)}/{population_size} initial candidates.")

    logging.info(f"Successfully generated {len(initial_population_jsons)} initial candidates.")
    return initial_population_jsons


# -------------------------------------------------------------------------
# [保留] 您原有的评估系统 (无需修改)
# -------------------------------------------------------------------------
class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.sacs_project_path = config.get('sacs.project_path')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.modifier = SacsFileModifier(self.sacs_project_path)
        self.runner = SacsRunner(project_path=self.sacs_project_path, sacs_install_path=config.get('sacs.install_path'))
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}
        # 计算基线指标用于动态归一化（若可用）
        self.baseline_weight_tonnes = None
        try:
            base_weight_res = calculate_sacs_weight_from_db(self.sacs_project_path)
            if base_weight_res.get('status') == 'success':
                self.baseline_weight_tonnes = max(1e-6, float(base_weight_res['total_weight_tonnes']))
                self.logger.info(f"动态归一化: 基线重量 = {self.baseline_weight_tonnes:.3f} 吨")
        except Exception as e:
            self.logger.warning(f"无法读取基线重量用于动态归一化: {e}")

    def evaluate(self, items):
        invalid_num = 0
        if not items: return [], {"invalid_num": 0, "repeated_num": 0}

        for item in items:
            try:
                raw_value = item.value
                try:
                    json_match = re.search(r'{\s*"new_code_blocks":\s*{.*?}\s*}', raw_value, re.DOTALL)
                    if json_match: raw_value = json_match.group(0)
                    elif 'candidate' in raw_value: raw_value = raw_value.split('<candidate>', 1)[1].rsplit('</candidate>', 1)[0].strip()
                    modifications = json.loads(raw_value)
                    new_code_blocks = modifications.get("new_code_blocks")
                except (json.JSONDecodeError, IndexError, AttributeError) as e:
                    self.logger.warning(f"Could not parse candidate JSON: {raw_value[:200]}... Error: {e}")
                    self._assign_penalty(item, "Invalid JSON format from LLM")
                    invalid_num += 1
                    continue

                if not new_code_blocks or not isinstance(new_code_blocks, dict):
                    self._assign_penalty(item, "Invalid candidate structure (no new_code_blocks)")
                    invalid_num += 1
                    continue

                # 仅针对“几何优化”允许修改 JOINT_*（或配置声明的目标）；过滤掉 GRUP_/PGRUP_ 等截面相关卡
                allowed_keys = set()
                # 允许所有 JOINT_*（几何节点）
                allowed_keys.update([k for k in new_code_blocks.keys() if k.startswith('JOINT_')])
                # 若配置显式给出可优化节点/耦合节点，也加入白名单（健壮性）
                opt_joints = self.config.get('sacs.optimizable_joints', []) or []
                coupled_map = self.config.get('sacs.coupled_joints', {}) or {}
                for j in opt_joints:
                    parts = j.split()
                    if len(parts) == 2 and parts[0] == 'JOINT':
                        allowed_keys.add(f"JOINT_{parts[1]}")
                        if parts[1] in coupled_map:
                            allowed_keys.add(f"JOINT_{coupled_map[parts[1]]}")

                filtered_blocks = {k: v for k, v in new_code_blocks.items() if k in allowed_keys}

                if not self.modifier.replace_code_blocks(filtered_blocks):
                    self._assign_penalty(item, "SACS file modification failed")
                    invalid_num += 1
                    continue

                analysis_result = self.runner.run_analysis(timeout=300)
                if not analysis_result.get('success'):
                    error_msg = analysis_result.get('error', 'Unknown SACS execution error')
                    self.logger.warning(f"SACS analysis failed. Reason: {error_msg}")
                    self._assign_penalty(item, f"SACS_Run_Fail: {str(error_msg)[:100]}")
                    invalid_num += 1
                    continue

                weight_res = calculate_sacs_weight_from_db(self.sacs_project_path)
                uc_res = get_sacs_uc_summary(self.sacs_project_path)

                if not (weight_res.get('status') == 'success' and uc_res.get('status') == 'success'):
                    error_msg = f"W:{weight_res.get('error', 'OK')}|UC:{uc_res.get('message', 'OK')}"
                    self.logger.warning("Metric extraction failed after successful SACS run.")
                    self._assign_penalty(item, f"Metric_Extraction_Fail: {error_msg}")
                    invalid_num += 1
                    continue

                max_uc_overall = uc_res.get('max_uc', 999.0)
                is_feasible = max_uc_overall <= 1.0
                raw_results = {'weight': weight_res['total_weight_tonnes'], 'axial_uc_max': uc_res.get('axial_uc_max', 999.0), 'bending_uc_max': uc_res.get('bending_uc_max', 999.0)}
                penalized_results = self._apply_penalty(raw_results, max_uc_overall)
                transformed = self._transform_objectives(penalized_results)
                overall_score = 1.0 - np.mean(list(transformed.values()))
                results_dict = {'original_results': raw_results, 'transformed_results': transformed, 'overall_score': overall_score, 'constraint_results': {'is_feasible': 1.0 if is_feasible else 0.0, 'max_uc': max_uc_overall}}
                item.assign_results(results_dict)
            except Exception as e:
                self.logger.critical(f"Unhandled exception during evaluation: {e}", exc_info=True)
                self._assign_penalty(item, f"Critical_Eval_Error: {e}")
                invalid_num += 1
        return items, {"invalid_num": invalid_num, "repeated_num": 0}

    def _apply_penalty(self, results: dict, max_uc: float) -> dict:
        penalized_results = results.copy()
        if max_uc > 1.0:
            # 更柔和且随超限平方增长的惩罚，并做上限截断，避免过早塌缩
            violation = max_uc - 1.0
            penalty_factor = min(6.0, 1.0 + 3.0 * (violation ** 2))
            self.logger.warning(f"Infeasible design: max_uc={max_uc:.3f}. Applying penalty factor {penalty_factor:.2f}.")
            if self.obj_directions.get('weight') == 'min':
                penalized_results['weight'] *= penalty_factor
        return penalized_results

    def _assign_penalty(self, item, reason=""):
        penalty_score = 99999
        original = {obj: penalty_score if self.obj_directions[obj] == 'min' else -penalty_score for obj in self.objs}
        results = {'original_results': original, 'transformed_results': {obj: 1.0 for obj in self.objs}, 'overall_score': -1.0, 'constraint_results': {'is_feasible': 0.0, 'max_uc': 999.0}, 'error_reason': reason}
        item.assign_results(results)

    def _transform_objectives(self, penalized_results: dict) -> dict:
        transformed = {}
        uc_min, uc_max = 0.0, 1.0
        # 动态权重归一化：相对基线的比例，限制在 [0.5, 2.0] 范围
        if self.baseline_weight_tonnes:
            weight = penalized_results.get('weight', self.baseline_weight_tonnes)
            ratio = np.clip(weight / self.baseline_weight_tonnes, 0.5, 2.0)
            # 将 [0.5, 2.0] 线性映射到 [0, 1]
            weight_norm = (ratio - 0.5) / 1.5
        else:
            # 回退到固定区间
            w_min, w_max = 50.0, 5000.0
            weight = np.clip(penalized_results.get('weight', w_max), w_min, w_max)
            weight_norm = (weight - w_min) / (w_max - w_min)
        if self.obj_directions.get('weight') == 'min':
            transformed['weight'] = weight_norm
        else:
            transformed['weight'] = 1.0 - weight_norm

        axial_uc = np.clip(penalized_results.get('axial_uc_max', uc_max), uc_min, uc_max)
        if self.obj_directions.get('axial_uc_max') == 'min':
            transformed['axial_uc_max'] = (axial_uc - uc_min) / (uc_max - uc_min)
        else:
            transformed['axial_uc_max'] = (uc_max - axial_uc) / (uc_max - uc_min)

        bending_uc = np.clip(penalized_results.get('bending_uc_max', uc_max), uc_min, uc_max)
        if self.obj_directions.get('bending_uc_max') == 'min':
            transformed['bending_uc_max'] = (bending_uc - uc_min) / (uc_max - uc_min)
        else:
            transformed['bending_uc_max'] = (uc_max - bending_uc) / (uc_max - uc_min)

        for key, val in transformed.items():
            transformed[key] = np.clip(val, 0.0, 1.0)

        return transformed