
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.MOLLM import MOLLM

def main():
    parser = argparse.ArgumentParser(description='Run MOLLM for Stellarator Coil Optimization')
    parser.add_argument('--config', type=str, 
                        default='problem/stellarator_coil_gsco_lite/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluate existing results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Starting MOLLM for Stellarator Coil Optimization...")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    
    # Initialize and run
    # Note: MOLLM expects the config path string or dict.
    # It will use the 'evalutor_path' in the config to load the Evaluator.
    
    mollm = MOLLM(
        config=args.config,
        resume=args.resume,
        eval=args.eval,
        seed=args.seed
    )
    
    if args.eval:
        print("Starting Evaluation...")
        mollm.load_evaluate()
    else:
        print("Starting Optimization Loop...")
        mollm.run()

if __name__ == "__main__":
    main()
