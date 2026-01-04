import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Add project root to path to ensure imports work if needed
sys.path.append(os.getcwd())

# Mock Item class if not importable
try:
    from algorithm.base import Item
except ImportError:
    class Item:
        pass

def load_gsco_results(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract f_B and f_S
    # GSCO history is a list of dicts: {'f_B': ..., 'f_S': ...}
    f_B = [d['f_B'] for d in data]
    f_S = [d['f_S'] for d in data]
    return np.array(f_B), np.array(f_S)

def load_mollm_results(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # data['all_mols'] is a list of [Item, order]
    # or potentially just a list of Items depending on version, checking structure
    
    mols = data.get('all_mols', [])
    f_B = []
    f_S = []
    
    for entry in mols:
        if isinstance(entry, list) or isinstance(entry, tuple):
            item = entry[0]
        else:
            item = entry
            
        if hasattr(item, 'property'):
            # Check keys
            if 'f_B' in item.property and 'f_S' in item.property:
                val_fb = item.property['f_B']
                # Check for failure values
                if val_fb > 1e4: 
                    continue
                    
                f_B.append(val_fb)
                f_S.append(item.property['f_S'])
                
    return np.array(f_B), np.array(f_S)

def plot_comparison(gsco_fb, gsco_fs, mollm_fb, mollm_fs, output_file):
    plt.figure(figsize=(10, 6))
    
    # Plot GSCO trajectory
    plt.plot(gsco_fs, gsco_fb, 'k--', label='True-GSCO (Baseline)', alpha=0.7)
    plt.scatter(gsco_fs, gsco_fb, c='black', s=30, marker='x')
    
    # Plot MOLLM points
    plt.scatter(mollm_fs, mollm_fb, c='red', s=40, alpha=0.6, label='MOLLM (LLM-Guided)')
    
    # Identify Pareto frontier for MOLLM
    if len(mollm_fs) > 0:
        # Sort by f_S
        sorted_indices = np.argsort(mollm_fs)
        m_fs_sorted = mollm_fs[sorted_indices]
        m_fb_sorted = mollm_fb[sorted_indices]
        
        pareto_fs = []
        pareto_fb = []
        current_min_fb = float('inf')
        
        for fs, fb in zip(m_fs_sorted, m_fb_sorted):
            if fb < current_min_fb:
                pareto_fs.append(fs)
                pareto_fb.append(fb)
                current_min_fb = fb
                
        plt.plot(pareto_fs, pareto_fb, 'r-', linewidth=2, label='MOLLM Frontier')

    plt.yscale('log')
    plt.xlabel('Sparsity (f_S: Number of Active Cells)')
    plt.ylabel('Field Error (f_B) [T²m²]')
    plt.title('Comparison: True-GSCO vs MOLLM')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gsco', default='true_gsco_results.json', help='Path to GSCO json results')
    parser.add_argument('--mollm', required=True, help='Path to MOLLM pickle results')
    parser.add_argument('--output', default='comparison_plot.png', help='Output plot filename')
    
    args = parser.parse_args()
    
    print(f"Loading GSCO results from {args.gsco}...")
    gsco_fb, gsco_fs = load_gsco_results(args.gsco)
    print(f"  Found {len(gsco_fb)} points. Min f_B: {np.min(gsco_fb):.4e}")
    
    print(f"Loading MOLLM results from {args.mollm}...")
    mollm_fb, mollm_fs = load_mollm_results(args.mollm)
    print(f"  Found {len(mollm_fb)} points. Min f_B: {np.min(mollm_fb) if len(mollm_fb) > 0 else 'N/A'}")
    
    plot_comparison(gsco_fb, gsco_fs, mollm_fb, mollm_fs, args.output)

if __name__ == "__main__":
    main()
