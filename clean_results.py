
import pickle
import json
import sys
import os
import numpy as np
from collections import defaultdict

# Import evaluator for re-calculation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator
except ImportError:
    # Mock if not found, but we need it for re-eval
    print("Warning: Could not import evaluator. Physics re-calc will fail.")
    SimpleGSCOEvaluator = None

class ConfigWrapper:
    def __init__(self, data):
        self.data = data
    
    def get(self, key, default=None):
        if not isinstance(key, str):
            return default
        
        # Handle direct key access first
        if key in self.data:
            return self.data[key]
            
        keys = key.split('.')
        curr = self.data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr
    
    def __getitem__(self, key):
        return self.get(key)

def clean_file(file_path):
    print(f"Cleaning {file_path}...")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    all_mols = data.get("all_mols", [])
    if not all_mols:
        return

    # Setup evaluator if needed
    evaluator = None
    if SimpleGSCOEvaluator:
        # Create a dummy config base
        config_data = {
            'goals': ['f_B', 'f_S', 'I_max'],
            'plasma_boundary': {
                'wout_file': "/home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/vmecpp/calculations/wout_w7x.nc",
                'plas_n': 32
            },
            'coil_design': {
                'wf_nPhi': 12,
                'wf_nTheta': 12,
                'unit_current': 0.2,
                'winding_surface_expansion': 1.2,
                'use_background_field': True
            },
            'objective_ranges': {
                'f_B': [0.0, 50.0],
                'f_S': [0, 144],
                'I_max': [0.0, 1.0]
            }
        }
        
        # Try to use config from pickle if available
        if 'config' in data:
            # We should use the pickle config but ensure wout_file is absolute
            loaded_config = data['config']
            # If loaded_config is a dict, we can merge or use it. 
            # If it's a wrapper object from the pickle, we might need to extract data.
            if hasattr(loaded_config, 'data'):
                loaded_dict = loaded_config.data
            elif isinstance(loaded_config, dict):
                loaded_dict = loaded_config
            else:
                loaded_dict = {}
                
            # Update our base config with loaded values (preserving defaults if missing)
            # A deep merge would be better but for now let's just ensure critical paths
            # We'll use the loaded dict but force the wout_file path
            if loaded_dict:
                config_data.update(loaded_dict)
                
                # Ensure plasma_boundary exists and has wout_file
                if 'plasma_boundary' not in config_data:
                    config_data['plasma_boundary'] = {}
                
                # Force absolute path
                config_data['plasma_boundary']['wout_file'] = "/home/dataset-assist-0/MOLLM-main/problem/stellarator_vmec/vmecpp/calculations/wout_w7x.nc"

        # Wrap it
        config = ConfigWrapper(config_data)
        evaluator = SimpleGSCOEvaluator(config)

    cleaned_count = 0
    
    for i, entry in enumerate(all_mols):
        try:
            # Handle different structures
            if isinstance(entry, tuple): 
                item = entry[0]
            elif isinstance(entry, list): 
                item = entry[0]
            else:
                item = entry
                
            if not hasattr(item, 'value'):
                continue
                
            config_json = json.loads(item.value)
            cells = config_json.get("cells", [])
            
            # 1. Deduplicate cells (last write wins logic from evaluator)
            valid_cells_map = {}
            for cell in cells:
                if isinstance(cell, list) and len(cell) == 3:
                    phi, theta, state = cell
                elif isinstance(cell, dict):
                    phi = cell.get('phi', cell.get('phi_idx'))
                    theta = cell.get('theta', cell.get('theta_idx'))
                    state = cell.get('state', cell.get('polarity', 0))
                else:
                    continue
                
                # Validation
                if state not in [-1, 0, 1]: continue
                if state == 0: continue
                
                # Last write wins - just like evaluator logic
                valid_cells_map[(int(phi), int(theta))] = int(state)
            
            cleaned_cells = [[k[0], k[1], v] for k, v in valid_cells_map.items()]
            
            # Check if change occurred or if I_max is suspicious (indicative of error fallback)
            current_imax = 0.0
            if hasattr(item, 'property'):
                current_imax = item.property.get('I_max', 0.0)
            
            # 10.0 is the fallback value in evaluator.py
            is_suspicious = (current_imax >= 9.9) 
            
            config_json['cells'] = cleaned_cells
            new_value = json.dumps(config_json)
            
            if item.value != new_value or is_suspicious:
                cleaned_count += 1
                item.value = new_value
                
                # Re-evaluate physics to ensure consistency
                if evaluator:
                    current_array = evaluator.cells_to_segment_currents(cleaned_cells)
                    f_B = evaluator._evaluate_field_error(current_array)
                    f_S = len(cleaned_cells)
                    I_max = (np.max(np.abs(current_array)) / 1e6) if len(current_array) > 0 else 0.0
                    
                    # Update properties
                    if not hasattr(item, 'property'):
                        item.property = {}
                    
                    item.property['f_B'] = f_B
                    item.property['f_S'] = f_S
                    item.property['I_max'] = I_max
                    
                    if is_suspicious:
                         print(f"  Fixed suspicious I_max {current_imax} -> {I_max}")
                
        except Exception as e:
            # print(f"Error processing item {i}: {e}")
            pass

    print(f"  Cleaned {cleaned_count} individuals.")
    
    # Save back
    if cleaned_count > 0:
        backup_path = file_path + ".bak"
        # Only create backup if it doesn't exist to preserve original
        if not os.path.exists(backup_path):
            try:
                with open(backup_path, "wb") as f:
                    pickle.dump(data, f)
            except:
                pass
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print("  Saved updated file.")
    else:
        print("  No changes needed.")

if __name__ == "__main__":
    base_dir = "results/stellarator_coil_gsco_lite"
    import glob
    files = glob.glob(f"{base_dir}/**/*.pkl", recursive=True)
    
    for f in sorted(files):
        if "metrics" in f: continue
        clean_file(f)
