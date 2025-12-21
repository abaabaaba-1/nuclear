
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.base import Item

def inspect_results(save_path):
    if not os.path.exists(save_path):
        print(f"File not found: {save_path}")
        return

    print(f"Loading results from {save_path}...")
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check structure
    # data keys: 'history', 'init_pops', 'final_pops', 'all_mols', 'properties', 'evaluation', 'running_time'
    
    all_mols = data.get('all_mols', [])
    print(f"Total evaluated items: {len(all_mols)}")
    
    if len(all_mols) > 0:
        # Check the first few items for gradient hints
        print("\nChecking for Gradient Hints in stored items:")
        count_with_hints = 0
        for i, (item, order) in enumerate(all_mols):
            if hasattr(item, 'gradient_hints') and item.gradient_hints:
                count_with_hints += 1
                if count_with_hints <= 3:
                    print(f"Item {i} Hints: {item.gradient_hints}")
        
        print(f"\nItems with gradient hints: {count_with_hints}/{len(all_mols)}")
    
    # Check prompts in history
    history = data.get('history', None)
    if history and hasattr(history, 'prompts') and len(history.prompts) > 0:
        print("\nChecking Prompts for Gradient Hints injection:")
        # history.prompts is a list of lists of prompts (one per generation?)
        last_prompts = history.prompts[-1]
        if isinstance(last_prompts, list) and len(last_prompts) > 0:
            sample_prompt = last_prompts[0]
            if "[GRADIENT HINTS]" in sample_prompt:
                print("✅ Found '[GRADIENT HINTS]' in the prompt text.")
                # print snippet
                start = sample_prompt.find("[GRADIENT HINTS]")
                print(f"Snippet: {sample_prompt[start:start+200]}...")
            else:
                print("❌ '[GRADIENT HINTS]' NOT found in the prompt text.")
                print("Prompt beginning:", sample_prompt[:500])
        else:
             print("Prompts structure unexpected:", type(last_prompts))

if __name__ == "__main__":
    # Path construction based on config
    # save_dir: "./results/stellarator_coil_gsco_lite"
    # model: zgca,gemini-2.5-flash-nothinking -> subfolder "zgca,gemini-2.5-flash-nothinking"
    # file: f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl
    
    result_path = "results/stellarator_coil_gsco_lite/zgca,gemini-2.5-flash-nothinking/mols/f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.pkl"
    inspect_results(result_path)
