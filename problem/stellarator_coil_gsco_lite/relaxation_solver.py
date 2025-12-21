
import numpy as np
import logging
from sklearn.linear_model import Lasso
from typing import Dict, List, Tuple, Optional
import json

# Import the evaluator to reuse its matrix calculation
from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator

class RelaxationSolverTool:
    def __init__(self, config_path: str):
        """
        Tool for LLM to perform "Relaxation + Projection" optimization.
        """
        with open(config_path, 'r') as f:
            # We assume config is a dict (json or yaml loaded before passing, 
            # but here we load from path for standalone usage)
            import yaml
            config_raw = yaml.safe_load(f)
            
        # Helper to flatten config (consistent with other scripts)
        def flatten_config(cfg, parent_key='', sep='.'):
            items = []
            for k, v in cfg.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        self.config = flatten_config(config_raw)

        # Initialize Evaluator to get the Response Matrix
        # We assume the config is compatible with SimpleGSCOEvaluator
        # We need to flatten the config if it's nested, or the evaluator handles it.
        # SimpleGSCOEvaluator expects a dict (possibly nested).
        
        self.evaluator = SimpleGSCOEvaluator(self.config)
        self.logger = logging.getLogger("RelaxationSolverTool")
        
        # Initialize Gradient Helper to compute the matrix
        # This gives us self.evaluator.response_matrix (Cells -> B_n)
        self.evaluator._init_gradient_helper()
        
        if not self.evaluator.gradient_helper_initialized:
            raise RuntimeError("Failed to initialize gradient helper (Simsopt might be missing).")
            
        # Prepare Data for Lasso
        # Objective: minimize 0.5 * || B_target - A * x ||^2_W + alpha * ||x||_1
        # Where W is the integration weights (dS)
        
        # 1. Integration Weights
        # dS_flat contains the Jacobian (area element)
        # norm_factor contains 0.5 / (N_theta * N_phi)
        # We want to match f_B definition: f_B = 0.5 * sum( (B_total)^2 * dS ) / (N*N)
        # Let W_vec = dS_flat * (0.5 / (N*N))
        # Note: In Lasso, the standard form is (1 / (2 * n_samples)) * ||y - Xw||^2 + alpha * ||w||_1
        # We need to adapt our weights to fit sklearn or solve manually.
        # Sklearn Lasso supports sample_weight.
        
        self.dS_flat = self.evaluator.dS_flat
        self.norm_factor = self.evaluator.norm_factor # = 0.5 / (N*N)
        
        # self.weights = self.dS_flat * (1.0 / (self.evaluator.surf_plas.quadpoints_phi.size * self.evaluator.surf_plas.quadpoints_theta.size))
        # Note: norm_factor has 0.5. We remove 0.5 because Lasso formulation usually has 0.5 factor implicit or explicit.
        # Sklearn objective: (1 / (2 * n_samples)) * ||y - Xw||^2 + alpha * ||w||_1
        # Our f_B = 0.5 * sum( w_i * r_i^2 )
        # If we use sample_weight = dS_flat, Sklearn computes:
        # (1 / (2 * n_samples)) * sum( dS_i * r_i^2 )
        # This matches our f_B form almost exactly, except for the overall constant factor.
        # We can adjust 'alpha' to compensate for any scaling.
        
        self.A = self.evaluator.response_matrix # Shape (N_cells, N_grid_points)
        # Sklearn expects (n_samples, n_features). 
        # Our A maps cells (features) to grid_points (samples).
        # So we need A.T
        self.X = self.A.T # Shape (N_grid_points, N_cells)
        
        # Target B field (External field needs to be cancelled)
        # B_total = B_coil + B_ext
        # We want B_total ~ 0  =>  B_coil ~ -B_ext
        # So target y = -B_ext
        if self.evaluator.B_ext_n is not None:
            self.y = -self.evaluator.B_ext_n.flatten()
        else:
            self.y = np.zeros(self.X.shape[0])
            
        # Sample weights for integration
        self.sample_weights = self.dS_flat
        
        self.n_cells = self.X.shape[1]
        self.grid_shape = (self.evaluator.wf_nPhi, self.evaluator.wf_nTheta)

    def solve(self, alpha: float, threshold: float, mask_indices: List[int] = None) -> Dict:
        """
        Executes the "Relaxation + Projection" strategy.
        
        Args:
            alpha (float): L1 regularization strength (controls sparsity in relaxation).
                           Higher alpha -> fewer active loops.
            threshold (float): Cutoff value for projection. 
                               Currents with abs(x) < threshold are set to 0.
                               Others are set to +1 or -1 (or kept continuous if threshold is None?).
                               Here we assume we project to discrete {-1, 0, 1}.
            mask_indices (List[int]): Optional. List of cell indices to force to 0.
            
        Returns:
            Dict containing:
                - cells: List of active cells [{'phi':.., 'theta':.., 'state':..}]
                - metrics: {'f_B': float, 'f_S': int}
                - continuous_solution: List of float (optional)
        """
        # 1. Define Constraints (Mask)
        # Sklearn Lasso doesn't support forcing Coeffs to 0 directly easily without modifying X.
        # We can just zero out the columns in X for masked indices.
        
        X_active = self.X.copy()
        if mask_indices:
            # Set columns to 0 so they aren't used
            X_active[:, mask_indices] = 0.0
            
        # 2. Run Lasso (Relaxation)
        # alpha in sklearn is constant multiplying the L1 term.
        # If alpha is too small, it might not converge well or just give least squares.
        # If alpha is 0, use LinearRegression (but we want Lasso path usually).
        
        if alpha <= 1e-9:
            # Use Ridge or unregularized? Stick to Lasso with tiny alpha
            model = Lasso(alpha=1e-9, fit_intercept=False, max_iter=2000)
        else:
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
            
        # Fit
        # Note: We need to rescale alpha because sklearn averages the loss.
        # Our f_B is ~12.0. The sum of weights is Surface Area (~200m^2).
        # Sklearn objective: 1/(2*N) * sum(w * err^2) + alpha * |x|
        # We might need to scale alpha to be meaningful relative to f_B.
        # Let the LLM figure out the magnitude, or we provide a reasonable range.
        
        model.fit(X_active, self.y, sample_weight=self.sample_weights)
        
        continuous_currents = model.coef_ # Shape (N_cells,)
        
        # 3. Projection (Discretization)
        # We assume the goal is integer {-1, 0, 1}.
        # Strategy:
        #  - If x > threshold -> +1
        #  - If x < -threshold -> -1
        #  - Else -> 0
        
        active_cells = []
        discrete_currents = np.zeros_like(continuous_currents)
        
        for idx, val in enumerate(continuous_currents):
            # Enforce mask again (though Lasso should have handled it via zeroed X)
            if mask_indices and idx in mask_indices:
                continue
                
            phi = idx // self.grid_shape[1]
            theta = idx % self.grid_shape[1]
            
            state = 0
            if val > threshold:
                state = 1
            elif val < -threshold:
                state = -1
            
            if state != 0:
                active_cells.append({
                    "phi": int(phi),
                    "theta": int(theta),
                    "state": int(state) # +1 or -1
                })
                discrete_currents[idx] = state
                
        # 4. Evaluate Discrete Solution
        # We can use the evaluator's method, but we have the matrix here, so it's faster.
        # B_total = B_ext + A.T * x_discrete (Wait, X = A.T, so X * x)
        
        # Re-calculate f_B for the discrete solution
        # B_coil = X @ discrete_currents
        # B_total = B_coil + self.y (Wait, self.y is -B_ext. So B_total = B_coil - (-B_ext) ? No)
        # self.y was set to -B_ext.
        # Residual r = y - Xw = -B_ext - B_coil = -(B_ext + B_coil).
        # f_B ~ sum(w * r^2).
        # So we can just use the model prediction error if we use the same weights.
        
        # B_pred = X @ discrete_currents
        # residual = self.y - B_pred
        # f_B_val = 0.5 * np.sum(self.sample_weights * residual**2) / (self.sample_weights.size ? No)
        
        # Let's stick to evaluator's _evaluate_field_error logic to be 100% consistent
        # But that requires reconstruction of segments. 
        # Using the matrix approximation is consistent with the "Relaxation" view.
        # Let's verify scaling.
        # Evaluator: f_B = 0.5 * sum(B_n^2 * dS) / (N_theta*N_phi)
        # Here: residual = -B_total.
        # We want 0.5 * sum(residual^2 * dS) / (N_points)
        
        B_pred = self.X @ discrete_currents
        # B_total = B_pred + B_ext
        # self.y = -B_ext
        # So B_total = B_pred - (-self.y) = B_pred - self.y ?? No
        # y = -B_ext.
        # residual (from regression) = y - B_pred = -B_ext - B_coil = -(B_total)
        # So residual^2 = B_total^2. Correct.
        
        residual = self.y - B_pred
        # We need to apply the exact same normalization as evaluator
        # norm_factor = 0.5 / N_points
        
        f_B_discrete = np.sum(residual**2 * self.sample_weights) * self.norm_factor
        f_S_discrete = len(active_cells)
        
        result = {
            "cells": active_cells,
            "metrics": {
                "f_B": float(f_B_discrete),
                "f_S": int(f_S_discrete)
            },
            # "continuous_max": float(np.max(np.abs(continuous_currents)))
        }
        
        return result

# Example usage for testing
if __name__ == "__main__":
    # Assumes config.yaml is in the same directory or adjust path
    import os
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        tool = RelaxationSolverTool(config_path)
        print("Solver initialized.")
        res = tool.solve(alpha=0.001, threshold=0.1)
        print("Result:", res['metrics'])
        print("Active cells:", len(res['cells']))
    else:
        print(f"Config not found at {config_path}")
