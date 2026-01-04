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
    f_B = [d['f_B'] for d in data]
    f_S = [d['f_S'] for d in data]
    return np.array(f_B), np.array(f_S)

def load_mollm_results(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    mols = data.get('all_mols', [])
    f_B = []
    f_S = []
    
    for entry in mols:
        if isinstance(entry, list) or isinstance(entry, tuple):
            item = entry[0]
        else:
            item = entry
            
        if hasattr(item, 'property'):
            if 'f_B' in item.property and 'f_S' in item.property:
                val_fb = item.property['f_B']
                if val_fb > 1e4: 
                    continue
                f_B.append(val_fb)
                f_S.append(item.property['f_S'])
                
    return np.array(f_B), np.array(f_S)

def plot_comparison(gsco_data, mollm_exp_data, mollm_enhanced_data, mollm_warm_start_data, output_file):
    plt.figure(figsize=(10, 6))
    
    # Unpack data
    gsco_fb, gsco_fs = gsco_data
    m_exp_fb, m_exp_fs = mollm_exp_data
    m_enh_fb, m_enh_fs = mollm_enhanced_data
    
    # Plot GSCO trajectory
    plt.plot(gsco_fs, gsco_fb, 'k--', label='True-GSCO (Baseline)', alpha=0.5)
    
    # Plot MOLLM Experience
    plt.scatter(m_exp_fs, m_exp_fb, c='blue', s=40, alpha=0.6, marker='o', label='MOLLM (Experience)')
    
    # Plot MOLLM Enhanced
    plt.scatter(m_enh_fs, m_enh_fb, c='red', s=40, alpha=0.6, marker='^', label='MOLLM (Exp + Enhanced Prompt)')

    # Plot MOLLM Warm Start
    if mollm_warm_start_data:
        m_ws_fb, m_ws_fs = mollm_warm_start_data
        plt.scatter(m_ws_fs, m_ws_fb, c='green', s=60, alpha=0.8, marker='*', label='MOLLM (Two-Step Warm Start)')
    
    # Plot Pareto Frontiers
    def plot_frontier(fs, fb, color, style, label):
        if len(fs) == 0: return
        sorted_indices = np.argsort(fs)
        fs_sorted = fs[sorted_indices]
        fb_sorted = fb[sorted_indices]
        
        pareto_fs = []
        pareto_fb = []
        current_min_fb = float('inf')
        
        for s, b in zip(fs_sorted, fb_sorted):
            if b < current_min_fb:
                pareto_fs.append(s)
                pareto_fb.append(b)
                current_min_fb = b
        plt.plot(pareto_fs, pareto_fb, color=color, linestyle=style, linewidth=2)

    plot_frontier(m_exp_fs, m_exp_fb, 'blue', '-', None)
    plot_frontier(m_enh_fs, m_enh_fb, 'red', '-', None)
    if mollm_warm_start_data:
        plot_frontier(m_ws_fs, m_ws_fb, 'green', '-', None)

    plt.yscale('log')
    plt.xlabel('Sparsity (f_S: Number of Active Cells)')
    plt.ylabel('Field Error (f_B) [T²m²]')
    plt.title('Impact of Initialization and Prompts on Optimization')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gsco', required=True, help='Path to GSCO json results')
    parser.add_argument('--mollm_exp', required=True, help='Path to MOLLM Exp pickle')
    parser.add_argument('--mollm_enhanced', required=True, help='Path to MOLLM Enhanced pickle')
    parser.add_argument('--mollm_warm_start', required=False, help='Path to MOLLM Warm Start pickle')
    parser.add_argument('--output', default='comparison_plot_enhanced.png', help='Output plot filename')
    
    args = parser.parse_args()
    
    print("Loading datasets...")
    gsco_data = load_gsco_results(args.gsco)
    print(f"GSCO: {len(gsco_data[0])} points")
    
    mollm_exp_data = load_mollm_results(args.mollm_exp)
    print(f"MOLLM Exp: {len(mollm_exp_data[0])} points, Min f_B: {np.min(mollm_exp_data[0]):.4f}")
    
    mollm_enhanced_data = load_mollm_results(args.mollm_enhanced)
    print(f"MOLLM Enhanced: {len(mollm_enhanced_data[0])} points, Min f_B: {np.min(mollm_enhanced_data[0]):.4f}")

    mollm_warm_start_data = None
    if args.mollm_warm_start:
        mollm_warm_start_data = load_mollm_results(args.mollm_warm_start)
        print(f"MOLLM Warm Start: {len(mollm_warm_start_data[0])} points, Min f_B: {np.min(mollm_warm_start_data[0]):.4f}")
    
    plot_comparison(gsco_data, mollm_exp_data, mollm_enhanced_data, mollm_warm_start_data, args.output)

if __name__ == "__main__":
    main()
