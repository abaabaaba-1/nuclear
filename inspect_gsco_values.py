
import pickle
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inspect_values(pickle_path):
    if not os.path.exists(pickle_path):
        print(f"File not found: {pickle_path}")
        return

    print(f"Loading results from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    all_mols = data.get('all_mols', [])
    print(f"Total molecules: {len(all_mols)}")
    
    f_B = []
    f_S = []
    I_max = []
    
    for item, _ in all_mols:
        props = item.property
        if props:
            f_B.append(props.get('f_B', np.nan))
            f_S.append(props.get('f_S', np.nan))
            I_max.append(props.get('I_max', np.nan))
            
    f_B = np.array(f_B)
    f_S = np.array(f_S)
    I_max = np.array(I_max)
    
    print("-" * 30)
    print(f"f_B (Field Error) [T^2 m^2]:")
    print(f"  Min: {np.nanmin(f_B):.4e}")
    print(f"  Max: {np.nanmax(f_B):.4e}")
    print(f"  Mean: {np.nanmean(f_B):.4e}")
    print(f"  Count < 1e4: {np.sum(f_B < 1e4)}")
    
    print("-" * 30)
    print(f"f_S (Complexity) [count]:")
    print(f"  Min: {np.nanmin(f_S)}")
    print(f"  Max: {np.nanmax(f_S)}")
    print(f"  Mean: {np.nanmean(f_S):.2f}")
    
    print("-" * 30)
    print(f"I_max (Max Current) [MA]:")
    print(f"  Min: {np.nanmin(I_max):.4f}")
    print(f"  Max: {np.nanmax(I_max):.4f}")
    print(f"  Mean: {np.nanmean(I_max):.4f}")
    print("-" * 30)

    # Check gradient hints
    print("\nChecking Gradient Hints for top 5 candidates:")
    sorted_indices = np.argsort(f_B) # Sort by f_B (lower is better)
    for i in range(min(5, len(all_mols))):
        idx = sorted_indices[i]
        item = all_mols[idx][0]
        print(f"\nRank {i+1}: f_B={item.property['f_B']:.4e}, f_S={item.property['f_S']}")
        if item.gradient_hints:
            print("  Hints:")
            for hint in item.gradient_hints[:3]: # Show top 3 hints
                print(f"    - {hint}")
        else:
            print("  No gradient hints found.")

if __name__ == "__main__":
    result_path = "results/stellarator_coil_gsco_lite/zgca,gemini-2.5-flash-nothinking/mols/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl"
    inspect_values(result_path)
