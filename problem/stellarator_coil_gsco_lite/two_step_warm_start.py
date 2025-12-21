import numpy as np
import yaml
import json
import os
import sys
import logging
import argparse
from scipy.optimize import lsq_linear

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def flatten_config(config, parent_key='', sep='.'):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return flatten_config(config)

class TwoStepWarmStarter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.evaluator = SimpleGSCOEvaluator(self.config)
        
        # Initialize gradient helper to get matrices
        self.evaluator._init_gradient_helper()
        
        # Matrix Setup
        # Minimize || R_w * x + b_w ||^2
        # subject to -1 <= x <= 1
        
        # Weights
        dS = self.evaluator.dS_flat
        norm_factor = self.evaluator.norm_factor 
        self.w = dS * norm_factor
        
        self.R = self.evaluator.response_matrix 
        self.b = self.evaluator.B_ext_n.flatten() if self.evaluator.B_ext_n is not None else np.zeros(self.evaluator.n_grid_points)
        
        # Apply weights to least squares formulation
        # || sqrt(W) (Rx + b) ||^2
        self.sqrt_w = np.sqrt(self.w)
        
        # Prepare for lsq_linear: min || A x - y ||^2
        # A = sqrt(W) * R
        # y = - sqrt(W) * b
        
        # R shape: (Cells, Grid). We need (Grid, Cells) for A usually?
        # A x. R is (Cells, Grid). Wait.
        # R_matrix in evaluator: (n_cells, n_grid_points). 
        # B = R^T * I ?? No.
        # Let's check evaluator:
        # current_B_n += state * self.response_matrix[idx]
        # So B = sum(x_i * R_i).
        # B vector = R.T @ x
        # R (in evaluator) has shape (144, 1024). 
        # So B = R.T @ x. 
        # R.T is (1024, 144). 
        
        self.A = self.evaluator.response_matrix.T * self.sqrt_w[:, np.newaxis]
        self.y = - self.b * self.sqrt_w
        
        logging.info(f"Least Squares Problem prepared. A shape: {self.A.shape}")

    def solve_continuous(self):
        """
        Solve bounded least squares: -1 <= x <= 1
        """
        logging.info("Solving continuous bounded least squares...")
        
        n_vars = self.A.shape[1]
        lb = -np.ones(n_vars)
        ub = np.ones(n_vars)
        
        res = lsq_linear(self.A, self.y, bounds=(lb, ub), verbose=1)
        
        logging.info(f"Continuous solution found. Cost: {res.cost:.4f}, Optimality: {res.optimality:.4f}")
        return res.x

    def stochastic_round(self, x_cont, num_samples=10, noise_std=0.1):
        """
        Generate multiple discrete candidates based on continuous solution.
        Strategies:
        1. Deterministic Rounding (nearest integer)
        2. Probabilistic Rounding (x as probability)
        3. Thresholding
        """
        candidates = []
        max_cells = self.config.get('llm_constraints.max_active_cells', 60)
        
        # 1. Deterministic Top-K (No noise)
        indices = np.argsort(np.abs(x_cont))[::-1]
        top_k_indices = indices[:max_cells]
        x_det = np.zeros_like(x_cont)
        for idx in top_k_indices:
            x_det[idx] = np.round(x_cont[idx])
        candidates.append(self._make_candidate(x_det, "deterministic_topk"))
        
        # 2. Stochastic Top-K with Gaussian Noise
        for i in range(num_samples):
            # Add noise to continuous solution to encourage diversity
            noisy_x = x_cont + np.random.normal(0, noise_std, size=x_cont.shape)
            
            # Temporary list to hold (index, value, noisy_magnitude)
            temp_sample = []
            
            for j, val in enumerate(noisy_x):
                # Clamp probability-like interpretation or just use sign of noisy value
                # Strategy: Use sign of noisy value if magnitude > threshold?
                # Or just keep top K magnitude of noisy vector.
                
                # Simple strategy: Keep top K magnitude of noisy vector, set to -1/1 based on sign
                temp_sample.append((j, np.sign(val), abs(val)))
            
            # Sort by magnitude descending
            temp_sample.sort(key=lambda x: x[2], reverse=True)
            
            # Keep top K
            x_sample = np.zeros_like(x_cont)
            for k in range(min(len(temp_sample), max_cells)):
                idx, sign, mag = temp_sample[k]
                if mag > 0.01: # Filter out very small noise
                    x_sample[idx] = sign if sign != 0 else 1 # Default to 1 if 0?
            
            candidates.append(self._make_candidate(x_sample, f"stochastic_noise_{i}"))
            
        return candidates

    def _make_candidate(self, x_vec, method_name):
        """Convert vector to cell list format"""
        cells = []
        nPhi = self.evaluator.wf_nPhi
        nTheta = self.evaluator.wf_nTheta
        
        for idx, val in enumerate(x_vec):
            if abs(val) > 0.1: # Non-zero
                phi = idx // nTheta
                theta = idx % nTheta
                state = int(np.sign(val))
                cells.append([phi, theta, state])
        
        # Evaluate
        # Note: We need to use evaluator to get accurate score
        currents = self.evaluator.cells_to_segment_currents(cells)
        f_B = self.evaluator._evaluate_field_error(currents)
        f_S = len(cells)
        
        return {
            "method": method_name,
            "cells": cells,
            "f_B": f_B,
            "f_S": f_S
        }

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    starter = TwoStepWarmStarter(config_path)
    
    # 1. Solve Continuous
    x_cont = starter.solve_continuous()
    
    # 2. Rounding
    candidates = starter.stochastic_round(x_cont, num_samples=20)
    
    # 3. Sort and Save
    candidates.sort(key=lambda c: c['f_B'])
    
    logging.info("Top 5 candidates from Warm Start:")
    for i, c in enumerate(candidates[:5]):
        logging.info(f"Rank {i+1}: Method={c['method']}, f_B={c['f_B']:.4f}, f_S={c['f_S']}")
    
    # Save as JSON for MOLLM or GSCO to load
    output_file = "warm_start_seeds.json"
    with open(output_file, "w") as f:
        json.dump([c['cells'] for c in candidates], f) # Just save the cell config list
        
    logging.info(f"Saved {len(candidates)} seeds to {output_file}")

if __name__ == "__main__":
    main()
