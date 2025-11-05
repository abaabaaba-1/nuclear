import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_process_data(filepath):
    """
    Loads and processes data from a specified JSON file path into a Pandas DataFrame.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert the "results" list into a DataFrame for easy handling
        df = pd.DataFrame(data['results'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not a valid JSON.")
        return None
    except KeyError:
        print(f"Error: Could not find 'results' key in file {filepath}.")
        return None

def plot_comparison(df_mollm, df_baseline, x_axis, y_axis_list, titles):
    """
    Plots a comparison between the MolLM and Baseline models for specified metrics.

    Args:
    df_mollm (pd.DataFrame): Data for the MolLM model.
    df_baseline (pd.DataFrame): Data for the Baseline model.
    x_axis (str): The column name to use for the x-axis.
    y_axis_list (list): A list containing two y-axis metric names [metric1, metric2].
    titles (list): A list containing the titles for the two subplots [title1, title2].
    """
    if df_mollm is None or df_baseline is None:
        print("Cannot generate plot because data loading failed.")
        return

    # Create a figure with 1 row and 2 columns for the subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('MolLM vs. Baseline (GA) Performance Comparison', fontsize=20, y=1.02)

    # --- Plot for the first metric ---
    ax1 = axes[0]
    # Plot Baseline data
    ax1.plot(df_baseline[x_axis], df_baseline[y_axis_list[0]], marker='o', linestyle='--', label='Baseline (GA)')
    # Plot MolLM data
    ax1.plot(df_mollm[x_axis], df_mollm[y_axis_list[0]], marker='s', linestyle='-', label='Our Model (MolLM)')
    
    ax1.set_title(titles[0], fontsize=16)
    ax1.set_xlabel(f'{x_axis} (Number of Unique Designs Evaluated)', fontsize=12)
    ax1.set_ylabel(y_axis_list[0], fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot for the second metric ---
    ax2 = axes[1]
    # Plot Baseline data
    ax2.plot(df_baseline[x_axis], df_baseline[y_axis_list[1]], marker='o', linestyle='--', label='Baseline (GA)')
    # Plot MolLM data
    ax2.plot(df_mollm[x_axis], df_mollm[y_axis_list[1]], marker='s', linestyle='-', label='Our Model (MolLM)')

    ax2.set_title(titles[1], fontsize=16)
    ax2.set_xlabel(f'{x_axis} (Number of Unique Designs Evaluated)', fontsize=12)
    ax2.set_ylabel(y_axis_list[1], fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot to a file
    output_filename = 'model_comparison_plot_english.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    # Define file paths
    base_dir = './moo_results/zgca,gemini-2.5-flash-nothinking/results/'
    mollm_file = os.path.join(base_dir, 'weight_axial_uc_max_bending_uc_max_sacs_expanded_3_obj_42.json')
    baseline_file = os.path.join(base_dir, 'weight_axial_uc_max_bending_uc_max_sacs_expanded_3_obj_baseline_GA_42.json')
    
    # --- Data Loading ---
    print("Loading MolLM model data...")
    df_mollm = load_and_process_data(mollm_file)
    
    print("Loading baseline model data...")
    df_baseline = load_and_process_data(baseline_file)
    
    # --- Plotting ---
    # Define the x-axis and y-axis columns for plotting
    x_axis_col = 'all_unique_moles'
    y_axis_cols = ['hypervolume', 'avg_top1']
    plot_titles = ['Hypervolume Comparison', 'Average Top-1 Performance (avg_top1) Comparison']
    
    print("Generating comparison plot...")
    plot_comparison(df_mollm, df_baseline, x_axis_col, y_axis_cols, plot_titles)
