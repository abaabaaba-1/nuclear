import numpy as np
import yaml
import json
import os
import sys
import logging
import argparse
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

class ContinuousOptimizer:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.evaluator = SimpleGSCOEvaluator(self.config)
        
        # Initialize gradient helper to get matrices
        self.evaluator._init_gradient_helper()
        
        # Matrix Setup
        # f_B = 0.5 * sum((R*x + b)^2 * dS) * norm_factor
        # Let's vectorize this.
        # W is a diagonal matrix of weights: dS * norm_factor * 0.5
        # We absorb 0.5 into the weights? 
        # Evaluator: f_B = 0.5 * np.sum(B_n_sq_matrix * dS) / (ntheta * nphi)
        # So weights w_j = 0.5 * dS_flat[j] / (ntheta * nphi)
        
        self.R = self.evaluator.response_matrix # (N_cells, N_grid)
        self.b = self.evaluator.B_ext_n.flatten() if self.evaluator.B_ext_n is not None else np.zeros(self.evaluator.n_grid_points)
        
        # Weights
        dS = self.evaluator.dS_flat
        norm_factor = self.evaluator.norm_factor # This includes 0.5 and 1/(N*N)
        # Wait, evaluator code: 
        # self.norm_factor = 0.5 / (self.surf_plas.quadpoints_phi.size * self.surf_plas.quadpoints_theta.size)
        # f_B = np.sum(B_n**2 * dS) * self.norm_factor
        
        self.w = dS * norm_factor
        
        # Precompute quadratic form matrices for speed
        # f_B = sum_j w_j * (sum_i R_ji x_i + b_j)^2
        #     = sum_j w_j * [ (sum_i R_ji x_i)^2 + 2 b_j (sum_i R_ji x_i) + b_j^2 ]
        #     = x^T (R^T W R) x + 2 (b^T W R) x + const
        
        # W_diag = np.diag(self.w) # Too big? N_grid is roughly 32*32=1024. Not too big.
        # But let's use broadcasting for speed.
        
        # R_weighted = R * sqrt(w)
        # b_weighted = b * sqrt(w)
        # Then f_B = || R_weighted x + b_weighted ||^2
        
        self.sqrt_w = np.sqrt(self.w)
        self.R_w = self.R * self.sqrt_w[np.newaxis, :] # Broadcasting? No, R is (Cells, Grid). We want to scale columns.
        self.R_w = self.R * self.sqrt_w[np.newaxis, :] 
        self.b_w = self.b * self.sqrt_w
        
        # Q = R_w @ R_w.T  (Shape: Cells x Cells) - Small! 144x144
        self.Q = self.R_w @ self.R_w.T
        self.c = self.R_w @ self.b_w
        self.const_term = np.sum(self.b_w**2)
        
        logging.info(f"Quadratic form prepared. Q shape: {self.Q.shape}")

    def objective(self, x, lambda_reg=0.0):
        """
        Objective function: f_B(x) + lambda * Penalty(x)
        Penalty forces x towards integers {-1, 0, 1}
        Penalty = sum (dist(x, nearest_int))^2
        """
        # 1. f_B
        # f_B = x^T Q x + 2 c^T x + const
        f_B = x @ self.Q @ x + 2 * self.c @ x + self.const_term
        
        # 2. Integer Penalty
        # Soft penalty: (x - 0)^2 * (x - 1)^2 * (x + 1)^2 ?
        # Or simpler: sin^2(pi * x)
        # Or distance to nearest integer. 
        # Using sin^2 is differentiable and periodic.
        # penalty = sum(sin(pi * x)^2)
        
        f_reg = 0.0
        grad_reg = np.zeros_like(x)
        
        if lambda_reg > 0:
             # We want to force -1, 0, 1.
             # sin(pi*x) is 0 at integer values.
             # But we also want to bound it. We will use bounds in the optimizer.
             sin_pi_x = np.sin(np.pi * x)
             f_reg = np.sum(sin_pi_x**2)
             
             # Gradient of sin^2(u) is 2*sin(u)*cos(u)*u' = sin(2u)*u'
             grad_reg = np.pi * np.sin(2 * np.pi * x)
             
        total_loss = f_B + lambda_reg * f_reg
        
        # Gradient
        # Grad f_B = 2 Q x + 2 c
        grad_f_B = 2 * self.Q @ x + 2 * self.c
        
        total_grad = grad_f_B + lambda_reg * grad_reg
        
        return total_loss, total_grad

    def solve_end_to_end(self, steps=10):
        """
        Gradually increase lambda_reg to force integer solutions.
        """
        # Initial guess: all zeros
        x0 = np.zeros(self.Q.shape[0])
        
        # Bounds: [-1, 1]
        bounds = [(-1.0, 1.0) for _ in range(len(x0))]
        
        lambdas = np.linspace(0, 1.0, steps) # Adjust scale as needed. f_B is around 12.
        # We need penalty to be significant.
        
        current_x = x0
        
        print(f"{'Step':<5} | {'Lambda':<10} | {'f_B':<10} | {'Sparsity':<10} | {'IntegerErr':<10}")
        print("-" * 60)
        
        for i, lam in enumerate(lambdas):
            # Scale lambda. If f_B ~ 10, and penalty ~ N_cells (144), we need lambda ~ 0.1 to 1.
            # Let's be aggressive later.
            lam_scaled = lam * 10.0 
            
            res = minimize(
                fun=self.objective,
                x0=current_x,
                args=(lam_scaled,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds
            )
            
            current_x = res.x
            
            # Metrics
            f_B_val = current_x @ self.Q @ current_x + 2 * self.c @ current_x + self.const_term
            # Sparsity: count > 0.1
            sparsity = np.sum(np.abs(current_x) > 0.1)
            # Integer error
            dist_int = np.sum(np.min([np.abs(current_x), np.abs(current_x-1), np.abs(current_x+1)], axis=0))
            
            print(f"{i:<5} | {lam_scaled:<10.4f} | {f_B_val:<10.4f} | {sparsity:<10} | {dist_int:<10.4f}")
            
        return current_x

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    optimizer = ContinuousOptimizer(config_path)
    
    logging.info("Starting End-to-End Optimization...")
    final_x = optimizer.solve_end_to_end(steps=20)
    
    # Rounding
    rounded_x = np.round(final_x)
    
    # Evaluate Final
    f_B_final = rounded_x @ optimizer.Q @ rounded_x + 2 * optimizer.c @ rounded_x + optimizer.const_term
    sparsity = np.sum(np.abs(rounded_x) > 0.5)
    
    logging.info(f"Final Rounded Result: f_B = {f_B_final:.4f}, Sparsity = {sparsity}")
    
    # Save to file for inspection
    output = {
        "final_continuous": final_x.tolist(),
        "final_rounded": rounded_x.tolist(),
        "f_B": f_B_final
    }
    with open("end_to_end_result.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
