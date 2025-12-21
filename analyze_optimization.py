
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import seaborn as sns

def analyze_json_logs():
    base_dir = "results/stellarator_coil_gsco_lite/baselines"
    files = glob.glob(f"{base_dir}/*_metrics.json")
    
    data = []
    for f in files:
        algo = os.path.basename(f).split('_')[0]
        seed = os.path.basename(f).split('_')[1]
        
        try:
            with open(f, 'r') as fp:
                content = json.load(fp)
                results = content.get('results', [])
                
                if not results:
                    continue
                    
                df = pd.DataFrame(results)
                if 'generated_num' not in df.columns:
                    df['generated_num'] = df.index * 50
                
                # Get final metrics
                final_row = df.iloc[-1]
                data.append({
                    'Algorithm': algo,
                    'Seed': seed,
                    'Evaluations': final_row.get('generated_num', 0),
                    'Final_HV': final_row.get('hypervolume', 0),
                    'Final_Top1': final_row.get('avg_top1', 0),
                    'Initial_Top1': df.iloc[0].get('avg_top1', 0),
                    'Improvement': final_row.get('avg_top1', 0) - df.iloc[0].get('avg_top1', 0)
                })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    summary_df = pd.DataFrame(data)
    print("Optimization Summary:")
    print(summary_df.sort_values(['Algorithm', 'Seed']))
    
    return summary_df

def inspect_pickle_values():
    """Inspect actual objective values from pickle files to see component breakdown"""
    import pickle
    
    base_dir = "results/stellarator_coil_gsco_lite/baselines"
    pkl_files = glob.glob(f"{base_dir}/*.pkl")
    
    print("\nDetailed Objective Breakdown (Sample from last generation):")
    
    for f in sorted(pkl_files):
        if "StandardGA_46" not in f and "SimulatedAnnealing_46" not in f: 
            # Just check a couple of representative files to save time/output
            if "StandardGA_42" not in f: continue
            
        print(f"\nFile: {os.path.basename(f)}")
        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
                all_mols = data.get('all_mols', [])
                if not all_mols:
                    print("  No molecules found.")
                    continue
                
                # Look at top 5 items by overall score (if available) or just last few
                # We need to reconstruct score since it might not be stored directly or easy to access
                # But we can look at f_B, f_S, I_max
                
                items_data = []
                for entry in all_mols:
                    item = entry[0] if isinstance(entry, (list, tuple)) else entry
                    if hasattr(item, 'property'):
                        props = item.property
                        items_data.append(props)
                
                df_items = pd.DataFrame(items_data)
                if df_items.empty:
                    print("  No property data found.")
                    continue
                    
                # Filter outliers
                df_valid = df_items[df_items['f_B'] < 100]
                
                if df_valid.empty:
                    print("  No valid items (f_B < 100).")
                    continue
                
                print("  Statistics for valid items (f_B < 100):")
                print(df_valid[['f_B', 'f_S', 'I_max']].describe())
                
                # Check min f_B
                min_fb = df_valid['f_B'].min()
                print(f"  Best f_B: {min_fb}")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    analyze_json_logs()
    inspect_pickle_values()
