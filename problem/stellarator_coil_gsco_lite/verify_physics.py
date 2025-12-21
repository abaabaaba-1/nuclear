
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator
import yaml

def verify_physics():
    print("Verifying Physics Calculations...")
    
    # Load config
    config_path = "problem/stellarator_coil_gsco_lite/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Helper to flatten config (copied from continuous_end_to_end.py)
    def flatten_config(config, parent_key='', sep='.'):
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_config(config)
    evaluator = SimpleGSCOEvaluator(flat_config)
    
    # 1. Verify Surface Area Calculation
    # The evaluator uses dS (Jacobian) * (1/N^2) for integration.
    # Let's check if this sums to the correct surface area.
    
    normal_vec = evaluator.surf_plas.normal()
    dS = np.sqrt(np.sum(normal_vec**2, axis=2))
    
    # Simsopt area() method
    simsopt_area = evaluator.surf_plas.area()
    
    # Manual integration
    nphi = evaluator.surf_plas.quadpoints_phi.size
    ntheta = evaluator.surf_plas.quadpoints_theta.size
    norm_factor = 1.0 / (nphi * ntheta)
    
    manual_area = np.sum(dS) * norm_factor
    
    print(f"Simsopt Surface Area: {simsopt_area:.6f} m^2")
    print(f"Manual Integration:   {manual_area:.6f} m^2")
    
    if np.isclose(manual_area, simsopt_area, rtol=1e-5):
        print("✅ Surface integration weights are CORRECT.")
    else:
        print("❌ Surface integration weights mismatch!")
        
    # 2. Verify f_B calculation consistency
    # Evaluator formula: 0.5 * sum(B_n^2 * dS) * norm_factor
    # Paper formula: 0.5 * integral((B.n)^2 dA)
    
    print("\nVerifying f_B consistency...")
    # Create a dummy current distribution
    cells = [[0, 0, 1]] # One loop
    currents = evaluator.cells_to_segment_currents(cells)
    f_B = evaluator._evaluate_field_error(currents)
    
    print(f"Calculated f_B for single loop: {f_B:.6e} T^2m^2")
    
    # Verify units roughly
    # B ~ mu0 * I / (2*pi*r)
    # I = 0.2 MA = 2e5 A. r ~ 0.3m.
    # B ~ 4e-7 * 2e5 / 2 = 0.04 T.
    # f_B ~ 0.5 * B^2 * Area ~ 0.5 * 0.0016 * 200 ~ 0.16.
    # If the value is around 0.01 - 1.0, it's reasonable.
    # If it's 1e-12 or 1e10, it's wrong.
    
    if 1e-4 < f_B < 100.0:
        print(f"✅ f_B magnitude ({f_B:.4f}) is physically reasonable.")
    else:
        print(f"⚠️ f_B magnitude ({f_B:.4f}) seems suspicious. Check units (MA vs A).")
        
    # 3. Verify Relaxation Solver Import and Initialization
    print("\nVerifying Relaxation Solver...")
    try:
        from problem.stellarator_coil_gsco_lite.relaxation_solver import RelaxationSolverTool
        solver = RelaxationSolverTool(config_path)
        print("✅ RelaxationSolverTool initialized successfully.")
        
        # Try a quick solve
        print("Running Lasso solve...")
        result = solver.solve(alpha=1e-5, threshold=0.5)
        print(f"✅ Solver result: f_B={result['metrics']['f_B']:.4f}, f_S={result['metrics']['f_S']}")
        print(f"   Active cells: {len(result['cells'])}")
        
    except Exception as e:
        print(f"❌ Relaxation Solver failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_physics()
