#!/usr/bin/env python3
"""Convenience launcher to run all baselines sequentially.

Usage:
    # Run all VMEC baselines
    python run_all_baselines.py --problem vmec --seed 123
    
    # Run all Coil baselines
    python run_all_baselines.py --problem coil --seed 123
    
    # Run specific baselines with multiple seeds
    python run_all_baselines.py --problem vmec --baselines ga nsga2 --seeds 42 43 44
    python run_all_baselines.py --problem coil --baselines ga nsga2 sms moead --seeds 70
    
    # Run MOLLM framework
    python run_all_baselines.py --problem coil --baselines mollm --seeds 42

This script assumes it is executed from the repo root and that the
MOLLM_env environment is already activated.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# VMEC baselines (plasma equilibrium optimization)
VMEC_BASELINES = {
    "ga": ["python", "baseline_ga.py", "problem/stellarator_vmec/config_ga.yaml"],
    "nsga2": ["python", "baseline_nsga2.py", "problem/stellarator_vmec/config_nsga2.yaml"],
    "sms": ["python", "baseline_sms.py", "problem/stellarator_vmec/config_sms.yaml"],
    "moead": ["python", "baseline_moead.py", "problem/stellarator_vmec/config_moead.yaml"],
    # LLM-based MOLLM baseline for VMEC (uses the standard VMEC config)
    "mollm": ["python", "main.py", "problem/stellarator_vmec/config.yaml"],
}

# Coil baselines (coil design optimization, original coil problem)
COIL_BASELINES = {
    "mollm": ["python", "main.py", "problem/stellarator_coil/config.yaml"],
    "ga": ["python", "baseline_ga.py", "problem/stellarator_coil/config_ga.yaml"],
    "nsga2": ["python", "baseline_nsga2.py", "problem/stellarator_coil/config_nsga2.yaml"],
    "sms": ["python", "baseline_sms.py", "problem/stellarator_coil/config_sms.yaml"],
    "moead": ["python", "baseline_moead.py", "problem/stellarator_coil/config_moead.yaml"],
}

# GSCO-Lite baselines (cell-based stellarator coil design, Hammond-style GSCO)
GSCO_LITE_BASELINES = {
    # LLM-based MOLLM optimizer on GSCO-Lite
    "mollm": ["python", "main.py", "problem/stellarator_coil_gsco_lite/config.yaml"],
    # Classical heuristics implemented in run_gsco_baselines.py
    "random": [
        "python",
        "run_gsco_baselines.py",
        "--algo",
        "random",
        "--config",
        "problem/stellarator_coil_gsco_lite/config.yaml",
    ],
    "ga": [
        "python",
        "run_gsco_baselines.py",
        "--algo",
        "ga",
        "--config",
        "problem/stellarator_coil_gsco_lite/config.yaml",
    ],
    "sa": [
        "python",
        "run_gsco_baselines.py",
        "--algo",
        "sa",
        "--config",
        "problem/stellarator_coil_gsco_lite/config.yaml",
    ],
    "greedy": [
        "python",
        "run_gsco_baselines.py",
        "--algo",
        "greedy",
        "--config",
        "problem/stellarator_coil_gsco_lite/config.yaml",
    ],
}

PROBLEM_CONFIGS = {
    "vmec": VMEC_BASELINES,
    "coil": COIL_BASELINES,
    "gsco_lite": GSCO_LITE_BASELINES,
}


def build_command(baselines_dict: dict, baseline_key: str, seed: int) -> list[str]:
    """Build command with seed argument."""
    base_cmd = baselines_dict[baseline_key]
    cmd = base_cmd.copy()
    cmd += ["--seed", str(seed)]
    return cmd


def run_baseline(baselines_dict: dict, baseline_key: str, seeds: list[int]) -> None:
    """Run a single baseline with multiple seeds."""
    for seed in seeds:
        cmd = build_command(baselines_dict, baseline_key, seed)
        print(f"\n{'='*70}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*70}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"{baseline_key} with seed {seed} failed (exit {proc.returncode})")
        print(f"✓ {baseline_key} (seed {seed}) completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline optimizers for stellarator problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all VMEC baselines with default seed
  python run_all_baselines.py --problem vmec
  
  # Run MOLLM for coil optimization
  python run_all_baselines.py --problem coil --baselines mollm --seeds 42
  
  # Run GA and NSGA-II for coil with multiple seeds
  python run_all_baselines.py --problem coil --baselines ga nsga2 --seeds 42 43 44
  
  # Run all coil baselines
  python run_all_baselines.py --problem coil
        """
    )
    parser.add_argument("--problem", choices=list(PROBLEM_CONFIGS.keys()), required=True,
                        help="Problem domain to optimize")
    parser.add_argument("--baselines", nargs="+", 
                        help="Baselines to run (default: all for selected problem)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seed list to sweep for each baseline (default: [42])")
    args = parser.parse_args()
    
    # Get baselines for selected problem
    baselines_dict = PROBLEM_CONFIGS[args.problem]
    
    # Use all baselines if not specified
    if args.baselines is None:
        baselines_to_run = list(baselines_dict.keys())
    else:
        # Validate baseline choices
        invalid = set(args.baselines) - set(baselines_dict.keys())
        if invalid:
            print(f"Error: Invalid baselines for {args.problem}: {invalid}")
            print(f"Available baselines: {list(baselines_dict.keys())}")
            sys.exit(1)
        baselines_to_run = args.baselines
    
    print(f"\n{'='*70}")
    print(f"Problem: {args.problem}")
    print(f"Baselines: {baselines_to_run}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*70}\n")
    
    for baseline_key in baselines_to_run:
        run_baseline(baselines_dict, baseline_key, args.seeds)
    
    print(f"\n{'='*70}")
    print(f"✓ All baselines completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
