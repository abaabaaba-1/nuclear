
import sys
import os
import yaml
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import json
import logging
import time
import argparse
from pathlib import Path
from copy import deepcopy

# Reuse the existing evaluator for setup and helper functions
from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrueGSCO")

from simsopt.geo import ToroidalWireframe
from simsopt.field import WireframeField

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value.get(k)
                if value is None:
                    return default
            return value
        except Exception:
            return default

class TrueGSCOSolver:
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.evaluator = SimpleGSCOEvaluator(self.config)
        
        # Grid dimensions
        self.nPhi = self.evaluator.wf_nPhi
        self.nTheta = self.evaluator.wf_nTheta
        self.n_cells = self.nPhi * self.nTheta
        
        # Forbidden cells
        self.forbidden_cells = set()
        forbidden_list = self.config.get('coil_design.forbidden_cells', [])
        if forbidden_list:
            for item in forbidden_list:
                if len(item) == 2:
                    self.forbidden_cells.add((item[0], item[1]))
        if self.forbidden_cells:
            logger.info(f"Forbidden cells: {len(self.forbidden_cells)} cells blocked")
        
        # Surface integration weights (dS)
        # evaluator._evaluate_field_error calculates: 0.5 * sum(B_n^2 * dS) / (ntheta * nphi)
        # We need to replicate this weighting
        self.surf = self.evaluator.surf_plas
        normal_vec = self.surf.normal()
        # dS is the magnitude of the normal vector (Jacobian)
        self.dS = np.sqrt(np.sum(normal_vec**2, axis=2)) 
        self.n_grid_points = self.dS.size
        # Flatten dS for vector operations
        self.dS_flat = self.dS.flatten()
        self.norm_factor = 0.5 / self.n_grid_points
        
        # Pre-calculated response matrices
        self.response_matrix = None # Shape: (n_cells, n_grid_points)
        self.B_background = None    # Shape: (n_grid_points,)
        
    def precompute_responses(self):
        """
        Pre-calculate the normal magnetic field response for each unit cell.
        This allows for rapid evaluation of candidates.
        """
        logger.info("Pre-computing cell response matrices (this may take a minute)...")
        start_time = time.time()
        
        # 1. Background field
        if self.evaluator.B_ext_n is not None:
            self.B_background = self.evaluator.B_ext_n.flatten()
        else:
            self.B_background = np.zeros(self.n_grid_points)
            
        # 2. Cell responses
        self.response_matrix = np.zeros((self.n_cells, self.n_grid_points))
        
        # We can batch this or do it one by one. 
        # Since we use simsopt, doing it one by one is safer but slower.
        # But we only do it once.
        
        for phi in range(self.nPhi):
            for theta in range(self.nTheta):
                idx = phi * self.nTheta + theta
                
                # Activate just this cell
                cell = [phi, theta, 1] 
                currents = self.evaluator.cells_to_segment_currents([cell])
                
                # Compute B_n on surface (without background)
                # We need to bypass the _evaluate_field_error method to get the vector B_n
                # So we manually call the simsopt routines
                
                wf = ToroidalWireframe(self.evaluator.surf_wf, self.nPhi, self.nTheta)
                wf.currents[:] = currents
                wf_field = WireframeField(wf)
                
                points = self.surf.gamma().reshape((-1, 3))
                wf_field.set_points(points)
                B_vec = wf_field.B()
                normals = self.surf.unitnormal().reshape((-1, 3))
                B_n = np.sum(B_vec * normals, axis=1)
                
                self.response_matrix[idx, :] = B_n
                
                if idx % 10 == 0:
                    print(f"  Computed response for cell {idx}/{self.n_cells}...", end='\r')
                    
        print(f"  Computed response for cell {self.n_cells}/{self.n_cells} [Done]")
        logger.info(f"Pre-computation finished in {time.time() - start_time:.2f}s")

    def solve(self, lambda_S=0.01, max_iter=100):
        """
        Run the Greedy Sparse Coil Optimization (GSCO) algorithm.
        
        f_GSCO = f_B + lambda_S * f_S
        """
        if self.response_matrix is None:
            self.precompute_responses()
            
        # Initial state: no active cells
        current_cells = {} # (phi, theta) -> polarity (-1 or 1)
        current_B_n = self.B_background.copy()
        
        # Initial objective
        # f_B = 0.5 * sum(B_n^2 * dS) / N
        f_B = np.sum(current_B_n**2 * self.dS_flat) * self.norm_factor
        f_S = 0
        f_current = f_B + lambda_S * f_S
        
        logger.info(f"Initial state: f_B={f_B:.4e}, f_S={f_S}, f_tot={f_current:.4e}")
        
        history = []
        
        for iteration in range(max_iter):
            best_delta_f = 0
            best_move = None # (phi, theta, new_polarity, new_B_n)
            
            # Iterate over all possible moves for all cells
            # Moves:
            # 1. Inactive -> +1 (Add)
            # 2. Inactive -> -1 (Add)
            # 3. +1 -> Inactive (Remove)
            # 4. -1 -> Inactive (Remove)
            # 5. +1 -> -1 (Flip)
            # 6. -1 -> +1 (Flip)
            
            # Simplified view: For each cell, try setting it to -1, 0, or 1
            # Calculate delta from current state
            
            for phi in range(self.nPhi):
                for theta in range(self.nTheta):
                    # Check forbidden
                    if (phi, theta) in self.forbidden_cells:
                        continue

                    idx = phi * self.nTheta + theta
                    current_pol = current_cells.get((phi, theta), 0)
                    response = self.response_matrix[idx]
                    
                    # Try all target polarities
                    for target_pol in [-1, 0, 1]:
                        if target_pol == current_pol:
                            continue
                            
                        # Calculate change in B_n
                        # current_B_n already includes current_pol * response
                        # new_B_n = current_B_n - current_pol * response + target_pol * response
                        delta_pol = target_pol - current_pol
                        new_B_n = current_B_n + delta_pol * response
                        
                        # Calculate new f_B
                        new_f_B = np.sum(new_B_n**2 * self.dS_flat) * self.norm_factor
                        
                        # Calculate new f_S
                        is_active = (target_pol != 0)
                        was_active = (current_pol != 0)
                        delta_S = int(is_active) - int(was_active)
                        new_f_S = f_S + delta_S
                        
                        new_f_total = new_f_B + lambda_S * new_f_S
                        
                        delta_f = new_f_total - f_current
                        
                        if delta_f < best_delta_f:
                            best_delta_f = delta_f
                            best_move = (phi, theta, target_pol, new_B_n, new_f_B, new_f_S, new_f_total)

            # Check if we found an improvement
            if best_move and best_delta_f < -1e-10:
                phi, theta, pol, new_B_n, new_f_B, new_f_S, new_f_total = best_move
                
                # Apply move
                if pol == 0:
                    del current_cells[(phi, theta)]
                else:
                    current_cells[(phi, theta)] = pol
                
                current_B_n = new_B_n
                f_B = new_f_B
                f_S = new_f_S
                f_current = new_f_total
                
                logger.info(f"Iter {iteration+1}: Best move cell({phi},{theta}) -> {pol}. "
                            f"dF={best_delta_f:.2e}. New f_B={f_B:.4e}, f_S={f_S}")
                
                history.append({
                    "iteration": iteration + 1,
                    "f_B": f_B,
                    "f_S": f_S,
                    "f_total": f_current,
                    "cells": [[k[0], k[1], v] for k, v in current_cells.items()]
                })
                
            else:
                logger.info(f"Converged at iteration {iteration}")
                break
                
        return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--lambda_S", type=float, default=1e-5, help="Sparsity penalty weight")
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--output", default="true_gsco_results.json", help="Output file")
    
    args = parser.parse_args()
    
    solver = TrueGSCOSolver(args.config)
    history = solver.solve(lambda_S=args.lambda_S, max_iter=args.max_iter)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Results saved to {args.output}")
