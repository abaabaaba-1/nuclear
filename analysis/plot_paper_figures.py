
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pickle
import glob
import pandas as pd

# Set style for publication quality
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.5

def load_multiple_json_metrics(filepaths):
    """
    Load metrics from multiple JSON files and aggregate them.
    Returns a DataFrame with index 'generated_num' and columns for mean/std of metrics.
    """
    all_runs = []
    
    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"Warning: File not found: {fp}")
            continue
            
        with open(fp, 'r') as f:
            try:
                data = json.load(f)
                results = data.get('results', [])
                if not results:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Ensure generated_num is present
                if 'generated_num' not in df.columns:
                    # Fallback if generated_num missing (e.g. use index * freq)
                    df['generated_num'] = df.index * 50 # Assumption based on log freq
                
                # Handle missing metric columns by filling with NaN or 0
                if 'hypervolume' not in df.columns:
                    df['hypervolume'] = 0.0
                if 'avg_top1' not in df.columns:
                     # Check if we can derive it or if it's named differently
                     # Some logs might use 'top1_auc' or similar, but let's stick to standard keys
                     df['avg_top1'] = 0.0

                all_runs.append(df)
            except Exception as e:
                print(f"Error loading {fp}: {e}")

    if not all_runs:
        return None

    # Align dataframes
    # We concat all and then group by generated_num
    combined = pd.concat(all_runs)
    
    # Group by generated_num and calculate mean/std
    # We round generated_num to nearest 10 or 50 to handle slight mismatches if any
    combined['generated_num_rounded'] = (combined['generated_num'] / 50).round() * 50
    
    grouped = combined.groupby('generated_num_rounded')
    mean_df = grouped.mean()
    std_df = grouped.std()
    count_df = grouped.count()
    
    return mean_df, std_df

def load_pkl_data(filepaths):
    """
    Load data from one or multiple pickle files.
    Returns a list of all items from all files.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
        
    all_items = []
    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"Warning: File not found: {fp}")
            continue
        try:
            with open(fp, 'rb') as f:
                data = pickle.load(f)
                items = data.get('all_mols', [])
                all_items.extend(items)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            
    return all_items

def plot_convergence(algo_files_map, metric='hypervolume', ylabel='Hypervolume', filename='convergence_hv.png', log_scale=False):
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("deep", len(algo_files_map))
    markers = ['o', 's', '^', 'D']
    
    for i, (algo_name, filepaths) in enumerate(algo_files_map.items()):
        # Handle single string vs list
        if isinstance(filepaths, str):
            filepaths = [filepaths]
            
        mean_df, std_df = load_multiple_json_metrics(filepaths)
        
        if mean_df is None:
            print(f"No data for {algo_name}")
            continue
            
        x = mean_df.index
        y = mean_df.get(metric, pd.Series(0, index=x))
        y_std = std_df.get(metric, pd.Series(0, index=x)) if std_df is not None else 0
        
        # Plot mean
        plt.plot(x, y, label=algo_name, color=colors[i], marker=markers[i%len(markers)], markersize=4, alpha=0.9)
        
        # Fill between for error bands (only if multiple runs exist and std > 0)
        if isinstance(y_std, pd.Series) and y_std.sum() > 0:
            plt.fill_between(x, y - y_std, y + y_std, color=colors[i], alpha=0.2)
        
    plt.xlabel('Number of Evaluations')
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    plt.title(f'{ylabel} Convergence')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")

def plot_pareto_front(pkl_files_map, filename='pareto_front.png'):
    plt.figure(figsize=(12, 5))
    
    # 1. f_B vs f_S
    plt.subplot(1, 2, 1)
    colors = sns.color_palette("deep", len(pkl_files_map))
    
    for i, (algo_name, filepaths) in enumerate(pkl_files_map.items()):
        all_items = load_pkl_data(filepaths)
        if not all_items:
            continue
            
        # Extract points
        fb = []
        fs = []
        
        for entry in all_items:
            # Handle tuple/list structure if present
            if isinstance(entry, (list, tuple)):
                item = entry[0]
            else:
                item = entry
                
            if not hasattr(item, 'property') or not item.property:
                continue
            
            # Filter huge errors
            val = item.property.get('f_B', float('inf'))
            if val < 100: 
                fb.append(val)
                fs.append(item.property.get('f_S', 0))
                
        plt.scatter(fs, fb, label=algo_name, color=colors[i], alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
    plt.xlabel('Complexity (Active Cells)')
    plt.ylabel('Field Error ($f_B$)')
    plt.yscale('log')
    plt.title('Solution Space ($f_B$ vs $f_S$)')
    plt.legend()
    
    # 2. f_B vs I_max
    plt.subplot(1, 2, 2)
    for i, (algo_name, filepaths) in enumerate(pkl_files_map.items()):
        all_items = load_pkl_data(filepaths)
        if not all_items:
            continue
            
        fb = []
        imax = []
        
        for entry in all_items:
            if isinstance(entry, (list, tuple)):
                item = entry[0]
            else:
                item = entry
                
            if not hasattr(item, 'property') or not item.property:
                continue
            val = item.property.get('f_B', float('inf'))
            if val < 100:
                fb.append(val)
                imax.append(item.property.get('I_max', 0))
        
        # Add jitter to x-axis (I_max) for better visualization of density
        # I_max values are discrete (0.0, 0.2, 0.4, etc.), so jitter helps separate them
        imax_jittered = np.array(imax) + np.random.uniform(-0.02, 0.02, size=len(imax))
                
        plt.scatter(imax_jittered, fb, label=algo_name, color=colors[i], alpha=0.5, s=25, edgecolors='none')

    plt.xlabel('Max Current ($I_{max}$) [MA]')
    plt.ylabel('Field Error ($f_B$)')
    plt.yscale('log')
    plt.title('Solution Space ($f_B$ vs $I_{max}$)\n(with x-jitter)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")

def main():
    base_dir = "results/stellarator_coil_gsco_lite"
    
    # Define file lists (Seeds 42, 43, 44, 45, 46)
    seeds = [42, 43, 44, 45, 46]
    
    # Baselines
    sa_json = [f"{base_dir}/baselines/SimulatedAnnealing_{s}_metrics.json" for s in seeds]
    ga_json = [f"{base_dir}/baselines/StandardGA_{s}_metrics.json" for s in seeds]
    rs_json = [f"{base_dir}/baselines/RandomSearch_{s}_metrics.json" for s in seeds]
    
    sa_pkl = [f"{base_dir}/baselines/SimulatedAnnealing_{s}.pkl" for s in seeds]
    ga_pkl = [f"{base_dir}/baselines/StandardGA_{s}.pkl" for s in seeds]
    rs_pkl = [f"{base_dir}/baselines/RandomSearch_{s}.pkl" for s in seeds]

    # MOLLM (Single seed 42 for now)
    mollm_json = [f"{base_dir}/zgca,gemini-2.5-flash-nothinking/results/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.json"]
    mollm_pkl = [f"{base_dir}/zgca,gemini-2.5-flash-nothinking/mols/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl"]
    
    json_files = {
        "MOLLM (LLM-GA)": mollm_json,
        "Standard GA": ga_json,
        "Simulated Annealing": sa_json,
        "Random Search": rs_json
    }
    
    pkl_files = {
        "MOLLM (LLM-GA)": mollm_pkl,
        "Standard GA": ga_pkl,
        "Simulated Annealing": sa_pkl,
        "Random Search": rs_pkl
    }
    
    print("Generating Hypervolume Convergence Plot...")
    plot_convergence(json_files, metric='hypervolume', ylabel='Hypervolume', filename='paper_convergence_hv.png')
    
    print("Generating Best Score Convergence Plot...")
    plot_convergence(json_files, metric='avg_top1', ylabel='Best Total Score', filename='paper_convergence_score.png')
    
    print("Generating Pareto Front Scatter Plot...")
    plot_pareto_front(pkl_files, filename='paper_pareto_front.png')

if __name__ == "__main__":
    main()
