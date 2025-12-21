
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def plot_pareto_front(pickle_path, output_file='gsco_pareto_front.png'):
    if not os.path.exists(pickle_path):
        print(f"File not found: {pickle_path}")
        return

    print(f"Loading results from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    all_mols = data.get('all_mols', [])
    print(f"Total molecules: {len(all_mols)}")
    
    # Extract objectives
    # properties are stored in item.property dictionary
    # keys: 'f_B', 'f_S', 'I_max'
    
    f_B = []
    f_S = []
    I_max = []
    
    for item, _ in all_mols:
        props = item.property
        if props:
            f_B.append(props.get('f_B', 1e5))
            f_S.append(props.get('f_S', 100))
            I_max.append(props.get('I_max', 10.0))
            
    if not f_B:
        print("No valid data found.")
        return

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: f_B vs f_S
    sc1 = axes[0].scatter(f_S, f_B, c=I_max, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('f_S (Complexity)')
    axes[0].set_ylabel('f_B (Field Error)')
    axes[0].set_title('Field Error vs Complexity')
    axes[0].set_yscale('log')
    plt.colorbar(sc1, ax=axes[0], label='I_max')
    
    # Plot 2: f_B vs I_max
    sc2 = axes[1].scatter(I_max, f_B, c=f_S, cmap='plasma', alpha=0.7)
    axes[1].set_xlabel('I_max (Max Current)')
    axes[1].set_ylabel('f_B (Field Error)')
    axes[1].set_title('Field Error vs Max Current')
    axes[1].set_yscale('log')
    plt.colorbar(sc2, ax=axes[1], label='f_S')

    # Plot 3: f_S vs I_max
    sc3 = axes[2].scatter(I_max, f_S, c=f_B, cmap='inferno_r', alpha=0.7) # reversed inferno so darker is better (lower f_B)
    axes[2].set_xlabel('I_max (Max Current)')
    axes[2].set_ylabel('f_S (Complexity)')
    axes[2].set_title('Complexity vs Max Current')
    cbar = plt.colorbar(sc3, ax=axes[2], label='f_B')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    result_path = "results/stellarator_coil_gsco_lite/zgca,gemini-2.5-flash-nothinking/mols/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl"
    plot_pareto_front(result_path)
