import numpy as np
import json
import logging
import random
import copy

# --- Import SACS-specific modules from the same directory ---
from .sacs_file_modifier import SacsFileModifier
from .sacs_runner import SacsAnalysisManager
from .sacs_interface_uc import get_sacs_uc_summary
from .sacs_interface_weight import calculate_sacs_volume
from .sacs_interface_ftg import get_sacs_fatigue_summary

# A baseline candidate to start from if generation fails.
# Extracted from your sacinp - 副本.txt file
BASELINE_CODE_BLOCKS = {
    "new_code_blocks": {
        "GRUP_LG1": "GRUP LG1         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.005.00",
        "GRUP_LG2": "GRUP LG2         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.006.15",
        "GRUP_LG3": "GRUP LG3         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.006.75",
        "GRUP_LG4": "GRUP LG4         42.000 1.375 29.0011.6050.00 1    1.001.00     0.500N490.00",
        "GRUP_T01": "GRUP T01         16.000 0.625 29.0111.2035.00 1    1.001.00     0.500N490.00",
        "GRUP_T02": "GRUP T02         20.000 0.750 29.0011.6035.00 1    1.001.00     0.500N490.00",
        "GRUP_T03": "GRUP T03         12.750 0.500 29.0111.6035.00 1    1.001.00     0.500N490.00",
        "PGRUP_P01": "PGRUP P01 0.3750I29.000 0.25036.000                                     490.0000"
    }
}


def _parse_and_modify_line(line, block_name):
    """A helper function to parse and apply a small random modification to a SACS line."""
    original_line = line
    try:
        if block_name.startswith("GRUP"):
            od_str = line[18:24]
            wt_str = line[25:30]
            od = float(od_str)
            wt = float(wt_str)

            # Apply random change to either OD or WT
            if random.choice([True, False]):  # Modify OD
                change_factor = random.uniform(0.9, 1.1)
                new_od = np.clip(od * change_factor, 10.0, 48.0)  # Using wide range for all GRUPs
                new_line = line[:18] + f"{new_od: >6.3f}" + line[24:]
            else:  # Modify WT
                change_factor = random.uniform(0.9, 1.1)
                new_wt = np.clip(wt * change_factor, 0.5, 2.5)  # Using wide range for all GRUPs
                new_line = line[:25] + f"{new_wt: >5.3f}" + line[30:]
            return new_line

        elif block_name.startswith("PGRUP"):
            thick_str = line[11:17]
            thick = float(thick_str)
            change_factor = random.uniform(0.9, 1.1)
            new_thick = np.clip(thick * change_factor, 0.2500, 0.7500)
            new_line = line[:11] + f"{new_thick:<6.4f}" + line[17:]
            return new_line

    except (ValueError, IndexError):
        return original_line  # Return original if parsing fails

    return original_line


def generate_initial_population(config, seed):
    """
    **CRITICAL FIX:** Generates an initial population with random variations.
    Instead of creating identical baseline candidates, this function now creates
    a diverse starting population by applying small, random modifications to
    the baseline SACS code for each individual. This is essential to break
    the cycle of the LLM receiving the same input repeatedly.
    """
    np.random.seed(seed)
    random.seed(seed)

    population_size = config.get('optimization.pop_size')
    initial_population = []

    for _ in range(population_size):
        # Create a deep copy to avoid modifying the original baseline
        new_candidate_blocks = copy.deepcopy(BASELINE_CODE_BLOCKS)

        # Select one random block to modify for this individual
        block_to_modify_key = random.choice(list(new_candidate_blocks["new_code_blocks"].keys()))
        block_to_modify_name = block_to_modify_key.replace("_", " ")  # e.g., GRUP_LG1 -> GRUP LG1
        original_sacs_line = new_candidate_blocks["new_code_blocks"][block_to_modify_key]

        # Apply a small random modification
        modified_sacs_line = _parse_and_modify_line(original_sacs_line, block_to_modify_name)
        new_candidate_blocks["new_code_blocks"][block_to_modify_key] = modified_sacs_line

        # Convert the modified dictionary to a JSON string and add to the population
        initial_population.append(json.dumps(new_candidate_blocks))

    # The first candidate can remain the pure baseline
    initial_population[0] = json.dumps(BASELINE_CODE_BLOCKS)

    return initial_population


class RewardingSystem:
    def __init__(self, config):
        self.config = config
        self.sacs_project_path = config.get('sacs.project_path')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize SACS tools
        self.modifier = SacsFileModifier(self.sacs_project_path)
        self.runner = SacsAnalysisManager(self.sacs_project_path)

        # Get objective names and directions from config
        self.objs = config.get('goals', [])
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

    def evaluate(self, items):
        """
        The main evaluation function.
        For each item (candidate solution), this function will:
        1. Parse the JSON string to get the new SACS code blocks.
        2. Apply these blocks to the SACS input file.
        3. Run the SACS analysis.
        4. Extract performance metrics (weight, uc, fatigue).
        5. Assign scores back to the item.
        """
        invalid_num = 0

        for item in items:
            try:
                # 1. Parse the candidate string (which is a JSON of code blocks)
                modifications = json.loads(item.value)
                new_code_blocks = modifications.get("new_code_blocks")

                if not new_code_blocks or not isinstance(new_code_blocks, dict):
                    self.logger.warning(f"Invalid candidate format: {item.value}")
                    self._assign_penalty(item, "Invalid JSON format")
                    invalid_num += 1
                    continue

                # 2. Apply modifications to SACS file
                if not self.modifier.replace_code_blocks(new_code_blocks):
                    self.logger.error("Failed to apply modifications to SACS file.")
                    self._assign_penalty(item, "File modification failed")
                    invalid_num += 1
                    continue

                # 3. Run SACS analysis
                analysis_result = self.runner.run_with_retry()
                if not analysis_result.get('success'):
                    self.logger.error(f"SACS analysis failed: {analysis_result.get('error', 'Unknown error')}")
                    self._assign_penalty(item, "SACS run failed")
                    invalid_num += 1
                    continue

                # 4. Extract performance metrics
                weight_res = calculate_sacs_volume(self.sacs_project_path)
                uc_res = get_sacs_uc_summary(self.sacs_project_path)
                ftg_res = get_sacs_fatigue_summary(self.sacs_project_path)

                # Check if all metrics were successfully extracted
                if not all([weight_res.get('status') == 'success',
                            uc_res.get('status') == 'success',
                            ftg_res.get('status') == 'success']):
                    self.logger.error("Failed to extract all performance metrics.")
                    self._assign_penalty(item, "Metric extraction failed")
                    invalid_num += 1
                    continue

                original = {
                    'weight': weight_res['total_volume_m3'],
                    'uc': uc_res['max_uc'],
                    'fatigue': ftg_res['min_life_years']
                }

                # 5. Transform and assign results
                transformed = self._transform_objectives(original)
                overall_score = len(self.objs) - np.sum(list(transformed.values()))  # Higher is better

                results = {
                    'original_results': original,
                    'transformed_results': transformed,
                    'overall_score': overall_score
                }
                item.assign_results(results)

            except Exception as e:
                self.logger.critical(f"Critical error during evaluation of an item: {e}", exc_info=True)
                self._assign_penalty(item, str(e))
                invalid_num += 1

        log_dict = {
            "invalid_num": invalid_num,
            "repeated_num": 0  # Repetition is handled by the framework
        }
        return items, log_dict

    def _assign_penalty(self, item, reason=""):
        """Assigns penalty scores to a failed candidate."""
        # Penalty is a high value for minimization objectives
        penalty_score = 999
        original = {obj: penalty_score for obj in self.objs}
        transformed = {obj: 1.0 for obj in self.objs}  # Normalized penalty
        overall_score = -penalty_score  # Very low fitness

        results = {
            'original_results': original,
            'transformed_results': transformed,
            'overall_score': overall_score,
            'error_reason': reason
        }
        item.assign_results(results)

    def _transform_objectives(self, original_results):
        """Normalizes and adjusts objectives for minimization."""
        transformed = {}

        # Weight (min, range approx. 2.5-4.0)
        # Normalize to [0, 1] for a typical range, e.g., 2.0 to 5.0
        w = original_results['weight']
        transformed['weight'] = np.clip((w - 2.0) / 3.0, 0, 1)

        # UC (min, range approx. 0.5-2.0)
        # Normalize to [0, 1] for a typical range, e.g., 0.5 to 2.5
        uc = original_results['uc']
        transformed['uc'] = np.clip((uc - 0.5) / 2.0, 0, 1)

        # Fatigue (max, range approx. 50-500)
        # Normalize and then invert for minimization
        ftg = original_results['fatigue']
        normalized_ftg = np.clip((ftg - 20) / 480.0, 0, 1)
        transformed['fatigue'] = 1 - normalized_ftg

        return transformed
