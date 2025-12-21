
import pickle
import json
import collections
import glob
import os

base_dir = "results/stellarator_coil_gsco_lite"
files = glob.glob(f"{base_dir}/**/*.pkl", recursive=True)

for file_path in files:
    print(f"Checking {file_path}...")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        all_mols = data.get("all_mols", [])
        if not all_mols:
            print("  No individuals found.")
            continue
            
        print(f"  Total individuals: {len(all_mols)}")

        dup_count = 0
        max_imax = 0.0
        
        for entry in all_mols:
            # entry is likely [Item, count] or just Item
            item = entry[0] if isinstance(entry, list) else entry
            
            try:
                # Some baselines might store item directly or differently
                if hasattr(item, 'value'):
                    config = json.loads(item.value)
                    cells = config.get("cells", [])
                    
                    # Check for duplicates
                    coords = []
                    for c in cells:
                        if isinstance(c, list):
                            coords.append((c[0], c[1]))
                        elif isinstance(c, dict):
                            coords.append((c.get('phi', c.get('phi_idx')), c.get('theta', c.get('theta_idx'))))
                    
                    counter = collections.Counter(coords)
                    has_dup = any(v > 1 for v in counter.values())
                    
                    if has_dup:
                        dup_count += 1
                        if dup_count == 1:
                            print(f"  Sample duplicate in {file_path}:")
                            print(f"  Cells: {cells}")
                            print(f"  Counter: {counter}")
                            print(f"  I_max: {item.property.get('I_max', 'N/A')}")
                        
                    if hasattr(item, 'property'):
                        imax = item.property.get('I_max', 0)
                        if imax > max_imax:
                            max_imax = imax
            except Exception as e:
                # print(f"  Error parsing item: {e}")
                pass

        print(f"  Individuals with duplicate cells: {dup_count}")
        print(f"  Max I_max found: {max_imax}")

    except Exception as e:
        print(f"  Failed to process: {e}")
