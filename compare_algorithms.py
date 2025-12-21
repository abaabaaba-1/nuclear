
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def get_convergence(all_mols):
    # Sort by order if possible, though all_mols is usually appended in order
    # all_mols is list of (item, gen/info)
    
    best_so_far = []
    current_best = float('inf')
    
    # Extract f_B values in order
    values = []
    for item, _ in all_mols:
        if not item or not item.property:
            continue
        val = item.property.get('f_B', float('inf'))
        values.append(val)
        
    # Compute running min
    for v in values:
        if v < current_best:
            current_best = v
        best_so_far.append(current_best)
        
    return best_so_far

def get_pareto_points(all_mols):
    # Return lists of f_B and f_S
    f_B = []
    f_S = []
    
    for item, _ in all_mols:
        if not item or not item.property:
            continue
        f_B.append(item.property.get('f_B', float('inf')))
        f_S.append(item.property.get('f_S', 0))
        
    return f_B, f_S

def main():
    base_dir = "results/stellarator_coil_gsco_lite"
    
    files = {
        "MOLLM (LLM-GA)": f"{base_dir}/zgca,gemini-2.5-flash-nothinking/mols/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl",
        "Standard GA": f"{base_dir}/baselines/StandardGA_42.pkl",
        "Simulated Annealing": f"{base_dir}/baselines/SimulatedAnnealing_42.pkl",
        "Random Search": f"{base_dir}/baselines/RandomSearch_42.pkl"
    }
    
    # Color map
    colors = {
        "MOLLM (LLM-GA)": "red",
        "Standard GA": "blue",
        "Simulated Annealing": "green",
        "Random Search": "gray"
    }
    
    plt.figure(figsize=(14, 6))
    
    # 1. Convergence Plot
    plt.subplot(1, 2, 1)
    
    for name, path in files.items():
        data = load_data(path)
        if data is None:
            continue
        
        all_mols = data.get('all_mols', [])
        convergence = get_convergence(all_mols)
        
        # Plot
        plt.plot(convergence, label=name, color=colors.get(name, 'black'), linewidth=2 if "MOLLM" in name else 1.5)
        print(f"{name}: Final Best f_B = {convergence[-1] if convergence else 'N/A'}")
        
    plt.xlabel('Evaluations')
    plt.ylabel('Best f_B (Field Error) [T^2 m^2]')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.yscale('log') # Log scale might be better if differences are large, or linear if close
    
    # 2. Pareto/Scatter Plot (Final Population or All Points)
    # Let's plot all points to see the distribution/exploration
    plt.subplot(1, 2, 2)
    
    for name, path in files.items():
        data = load_data(path)
        if data is None:
            continue
            
        all_mols = data.get('all_mols', [])
        fb, fs = get_pareto_points(all_mols)
        
        # Plot scatter
        plt.scatter(fs, fb, label=name, color=colors.get(name, 'black'), alpha=0.5, s=20, edgecolors='none')
        
    plt.xlabel('f_S (Complexity / Active Cells)')
    plt.ylabel('f_B (Field Error)')
    plt.title('Solution Space Exploration (f_B vs f_S)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    print("Comparison plot saved to algorithm_comparison.png")

if __name__ == "__main__":
    main()
