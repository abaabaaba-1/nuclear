# evaluator.py (Definitive Fix - Employs a robust, context-aware line modification strategy)
import numpy as np
import json
import logging
import random
import copy
import re
from pathlib import Path

# --- 【已修正】使用相对路径导入，以匹配您的项目结构 ---
from .sacs_file_modifier import SacsFileModifier
from .sacs_runner import SacsRunner
from .sacs_interface_uc import get_sacs_uc_summary
from .sacs_interface_weight_improved import calculate_sacs_weight_from_db
# --- 修正结束 ---

# REWRITTEN: _parse_and_modify_line - The core of the new robust solution.
# This function no longer uses separate get/build helpers. It performs a minimal, in-place modification.
def _parse_and_modify_line(original_line: str, block_name: str, config=None) -> str:
    """
    Parses and modifies a JOINT line using a robust, surgical replacement strategy
    that preserves the original file's exact formatting, including spacing and precision.
    """
    try:
        keyword = block_name.split()[0]
        if keyword != "JOINT":
            return original_line.rstrip()

        # 1. Find all numbers and their precise start/end positions in the string.
        # This pattern finds numbers with or without decimals, handling scientific notation.
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
        
        # Determine original precision
        if '.' in original_text:
            precision = len(original_text.split('.')[1])
        else:
            precision = 0
        
        # Create a dynamic format specifier: e.g., ">10.2f" for right-align, 10 chars, 2 decimals
        format_spec = f">{original_len}.{precision}f"
        new_text = format(new_value, format_spec)

        # Prevent overflow from breaking the format. Truncate if necessary.
        if len(new_text) > original_len:
            new_text = new_text[:original_len]

        # 6. Surgically replace the old number text with the new formatted text
        start, end = target_match.span()
        new_line = original_line[:start] + new_text + original_line[end:]
        
        return new_line.rstrip()

    except Exception as e:
        logging.error(f"CRITICAL error in _parse_and_modify_line for '{block_name}': {e}", exc_info=True)
        return original_line.rstrip()

# ADDED: A new, simple coordinate getter needed for coupled joints.
def _get_coords_from_modified_line(modified_line: str) -> dict:
    """A simple parser to get the first three float values from a line."""
    try:
        num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
        all_numbers = [float(n) for n in num_pattern.findall(modified_line)]
        if len(all_numbers) < 3: return None
        return {'x': all_numbers[0], 'y': all_numbers[1], 'z': all_numbers[2]}
    except Exception:
        return None

# ADDED: A new, robust builder for coupled (slave) joints.
def _build_slave_joint_line(original_slave_line: str, master_coords: dict) -> str:
    """
    Rebuilds a slave joint line using the coordinates from its master.
    It uses the same robust, format-preserving logic as _parse_and_modify_line.
    """
    try:
        num_pattern = re.compile(r'-?\d+\.\d*(?:[eE][-+]?\d+)?')
        matches = list(num_pattern.finditer(original_slave_line))
        if len(matches) < 3: return original_slave_line.rstrip()

        x_match, y_match, z_match = matches[0], matches[1], matches[2]
        
        # This list makes it easy to iterate and replace
        replacements = [
            {'match': x_match, 'value': master_coords['x']},
            {'match': y_match, 'value': master_coords['y']},
            {'match': z_match, 'value': master_coords['z']}
        ]
        
        # Apply replacements from right to left to not mess up indices
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
    sacs_file_path = config.get('sacs.project_path')
    if not sacs_file_path:
        logging.error("Config missing 'sacs.project_path'.")
        return {}

    # Using 'sacinp.demo06' to be consistent with SacsFileModifier.
    sacs_file = Path(sacs_file_path) / "sacinp.demo13"


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


SEED_BASELINE = {
    "new_code_blocks": {
        "GRUP_DUM": "GRUP DUM         12.000 1.000 29.0011.6036.00 9    1.001.00     0.500N490.00",
        "GRUP_LG6_1": "GRUP LG6         36.000 0.750 29.0011.0036.00 1    1.001.00     0.500N490.003.25",
        "GRUP_LG6_CONE": "GRUP LG6 CONE                 29.0011.6036.00 1    1.001.00     0.500N490.004.95",
        "GRUP_LG6_2": "GRUP LG6         26.000 0.750 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_LG7": "GRUP LG7         26.000 0.750 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_SHF": "GRUP SHF          4.000 1.000 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_SK2": "GRUP SK2 W8X24                29.0011.6036.00 1    1.001.00     0.500N1.00-2",
        "GRUP_SKD": "GRUP SKD W12X30               29.0011.6036.00 1    1.001.00     0.500N1.00-2",
        "GRUP_STB": "GRUP STB          6.000 1.000 29.0011.6036.00 9    1.001.00     0.500N1.00-2",
        "GRUP_VB1": "GRUP VB1         12.750 0.625 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_VB2": "GRUP VB2          8.825 0.500 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_VBS": "GRUP VBS         12.750 0.625 29.0011.6036.00 1    1.001.00     0.500N490.00",
        "GRUP_W01": "GRUP W01 W24X162              29.0111.2035.97 1    1.001.00     0.500 490.00",
        "GRUP_W02": "GRUP W02 W24X131              29.0111.2035.97 1    1.001.00     0.500 490.00",
        "PGRUP_P01": "PGRUP P01 0.3750I29.000 0.25036.000                                     490.0000",
        "PGRUP_PLT": "PGRUP PLT 0.2500 29.000 0.25036.000                                     490.0000"
    }
}

def generate_initial_population(config, seed):
    np.random.seed(seed); random.seed(seed)
    population_size = config.get('optimization.pop_size')
    optimizable_joints = config.get('sacs.optimizable_joints', [])
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
    initial_population_jsons.append(master_seed_str)
    seen_candidates.add(master_seed_str)
    logging.info(f"Starting generation of initial population of size {population_size}...")
    max_tries, try_count = population_size * 10, 0

    while len(initial_population_jsons) < population_size and try_count < max_tries:
        base_candidate = copy.deepcopy(master_seed)
        if not optimizable_joints:
            logging.error("No optimizable joints defined in config; cannot generate new candidates.")
            break

        num_modifications = random.randint(1, max(1, len(optimizable_joints) // 2))
        items_to_modify = random.sample(optimizable_joints, min(num_modifications, len(optimizable_joints)))

        for item_name in items_to_modify:
            item_key = item_name.replace(" ", "_")
            original_sacs_line = base_candidate["new_code_blocks"][item_key]
            # Use the new robust modification function
            modified_sacs_line = _parse_and_modify_line(original_sacs_line, item_name, config=config)
            base_candidate["new_code_blocks"][item_key] = modified_sacs_line

            if item_name.startswith("JOINT"):
                joint_id = item_name.split(" ")[1]
                if joint_id in coupled_joints_map:
                    slave_id = coupled_joints_map[joint_id]
                    slave_key = f"JOINT_{slave_id}"
                    # Get the new master coordinates and update the slave
                    master_coords = _get_coords_from_modified_line(modified_sacs_line)
                    if master_coords and slave_key in base_candidate["new_code_blocks"]:
                        original_slave_line = base_candidate["new_code_blocks"][slave_key]
                        # Use the new robust slave line builder
                        new_slave_line = _build_slave_joint_line(original_slave_line, master_coords)
                        base_candidate["new_code_blocks"][slave_key] = new_slave_line

        candidate_str = json.dumps(base_candidate, sort_keys=True)
        if candidate_str not in seen_candidates:
            initial_population_jsons.append(candidate_str)
            seen_candidates.add(candidate_str)
        try_count += 1

    if len(initial_population_jsons) < population_size:
        logging.warning(f"Only generated {len(initial_population_jsons)}/{population_size} initial candidates.")

    logging.info(f"Successfully generated {len(initial_population_jsons)} initial candidates.")
    return initial_population_jsons


class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.sacs_project_path = config.get('sacs.project_path')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.modifier = SacsFileModifier(self.sacs_project_path)
        self.runner = SacsRunner(project_path=self.sacs_project_path, sacs_install_path=config.get('sacs.install_path'))
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

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

                if not self.modifier.replace_code_blocks(new_code_blocks):
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
            penalty_factor = 1.0 + (max_uc - 1.0) * 5.0
            self.logger.warning(f"Infeasible design: max_uc={max_uc:.3f}. Applying penalty factor {penalty_factor:.2f}.")
            if self.obj_directions['weight'] == 'min':
                penalized_results['weight'] *= penalty_factor
        return penalized_results

    def _assign_penalty(self, item, reason=""):
        penalty_score = 99999
        original = {obj: penalty_score if self.obj_directions[obj] == 'min' else -penalty_score for obj in self.objs}
        results = {'original_results': original, 'transformed_results': {obj: 1.0 for obj in self.objs}, 'overall_score': -1.0, 'constraint_results': {'is_feasible': 0.0, 'max_uc': 999.0}, 'error_reason': reason}
        item.assign_results(results)

    def _transform_objectives(self, penalized_results: dict) -> dict:
        transformed = {}
        w_min, w_max, uc_min, uc_max = 50, 5000, 0.0, 1.0

        weight = np.clip(penalized_results.get('weight', w_max), w_min, w_max)
        if self.obj_directions.get('weight') == 'min':
            transformed['weight'] = (weight - w_min) / (w_max - w_min)
        else:
            transformed['weight'] = (w_max - weight) / (w_max - w_min)

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