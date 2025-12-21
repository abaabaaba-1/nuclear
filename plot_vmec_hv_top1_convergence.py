#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot HV and top-1 convergence curves from VMEC JSON result logs.

This script is tailored for the stellarator VMEC 3-objective experiments.
It expects JSON files with the following structure (as produced by MOLLM
and baseline scripts):

{
    "results": [
        {
            "generated_num": int,
            "hypervolume": float,
            "avg_top1": float,
            ...
        },
        ...
    ],
    "params": "..."
}

By default, it plots curves for the following files in the project root:
- volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_42.json
- volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_GA_optimized_41.json
- volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_NSGA2_40.json
- volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_SMSEMOA_42.json

The x-axis uses "generated_num" (total evaluated candidates) when
available; otherwise it falls back to the index of the result entry.

Output: a PNG figure with two subplots (HV and avg_top1) saved into
"moo_results/figure/vmec_3obj_hv_top1_convergence.png".
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent


def load_results(path: Path) -> Tuple[List[float], List[float], List[float]]:
    """Load a JSON result file and extract x, hv, top1 series.

    x: generated_num (or index fallback)
    hv: hypervolume
    top1: avg_top1
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("results", [])
    if not records:
        raise ValueError(f"No 'results' array found in {path}")

    xs: List[float] = []
    hvs: List[float] = []
    top1s: List[float] = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        hv = rec.get("hypervolume")
        top1 = rec.get("avg_top1")
        if hv is None or top1 is None:
            continue

        x = rec.get("generated_num")
        if x is None:
            # fall back to index if generated_num is not present
            x = idx

        xs.append(float(x))
        hvs.append(float(hv))
        top1s.append(float(top1))

    if not xs:
        raise ValueError(f"No valid hv/top1 records found in {path}")

    return xs, hvs, top1s


def pretty_label(path: Path) -> str:
    """Generate a short label from the file name."""
    name = path.stem
    if "baseline_GA" in name:
        return "Baseline GA"
    if "baseline_NSGA2" in name:
        return "Baseline NSGA-II"
    if "baseline_SMSEMOA" in name:
        return "Baseline SMSEMOA"
    # default: LLM-based run
    return "MOLLM (LLM-assisted)"


def main() -> None:
    import argparse

    default_files = [
        "volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_42.json",
        "volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_GA_optimized_41.json",
        "volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_NSGA2_40.json",
        "volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_baseline_SMSEMOA_42.json",
    ]

    parser = argparse.ArgumentParser(
        description="Plot HV and top1 convergence from VMEC JSON logs",
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        default=default_files,
        help="JSON result files to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: moo_results/figure/vmec_3obj_hv_top1_convergence.png)",
    )

    args = parser.parse_args()

    paths: List[Path] = []
    for p in args.json_files:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            print(f"[Skip] JSON not found: {path}")
            continue
        paths.append(path)

    if not paths:
        raise SystemExit("No valid JSON files to plot.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_hv, ax_top1 = axes

    for path in paths:
        try:
            xs, hvs, top1s = load_results(path)
        except Exception as e:
            print(f"[Skip] Failed to load {path.name}: {e}")
            continue

        label = pretty_label(path)
        ax_hv.plot(xs, hvs, marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax_top1.plot(xs, top1s, marker="o", markersize=2.5, linewidth=1.2, label=label)

    ax_hv.set_xlabel("Generated candidates")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV convergence")

    ax_top1.set_xlabel("Generated candidates")
    ax_top1.set_ylabel("avg_top1")
    ax_top1.set_title("Top-1 convergence (avg_top1)")

    ax_hv.legend(fontsize=8)
    ax_top1.legend(fontsize=8)

    fig.tight_layout()

    if args.out is not None:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_dir = PROJECT_ROOT / "moo_results" / "figure"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vmec_3obj_hv_top1_convergence.png"

    fig.savefig(out_path, dpi=150)
    print(f"Convergence figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
