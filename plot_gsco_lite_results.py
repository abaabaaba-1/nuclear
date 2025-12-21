#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot GSCO-Lite results for Random, SA, GA and MOLLM on seeds 42/43.

This script reads the existing JSON metrics files and produces simple
convergence and final-metric plots:

- For each seed (42, 43):
  - avg_top1 vs generated_num for all algorithms.
  - hypervolume vs generated_num for all algorithms.
- For each seed (42, 43):
  - Bar charts comparing the final avg_top1 and hypervolume across algorithms.

It only reads from the `results/` folder and writes PNGs to
`results/stellarator_coil_gsco_lite/plots/`.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "stellarator_coil_gsco_lite"
PLOTS_DIR = RESULTS_DIR / "plots"

# Known metrics files for this project
ALGO_FILES: Dict[str, Dict[int, Path]] = {
    "RandomSearch": {
        42: RESULTS_DIR / "baselines" / "RandomSearch_42_metrics.json",
        43: RESULTS_DIR / "baselines" / "RandomSearch_43_metrics.json",
    },
    "StandardGA": {
        42: RESULTS_DIR / "baselines" / "StandardGA_42_metrics.json",
        43: RESULTS_DIR / "baselines" / "StandardGA_43_metrics.json",
    },
    "SimulatedAnnealing": {
        42: RESULTS_DIR / "baselines" / "SimulatedAnnealing_42_metrics.json",
        43: RESULTS_DIR / "baselines" / "SimulatedAnnealing_43_metrics.json",
    },
    "GreedyGSCO": {
        42: RESULTS_DIR / "baselines" / "GreedyGSCO_42_metrics.json",
    },
    "MOLLM": {
        42: RESULTS_DIR
        / "zgca,gemini-2.5-flash-nothinking"
        / "results"
        / "f_B_f_S_I_max_gsco_lite_mollm_warm_start_42.json",
        43: RESULTS_DIR
        / "zgca,gemini-2.5-flash-nothinking"
        / "results"
        / "f_B_f_S_I_max_gsco_lite_mollm_warm_start_43.json",
    },
}


def load_curve(path: Path) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """Load (generated_num, avg_top1, hypervolume) curves from a metrics JSON.

    Returns None if the file is missing or malformed.
    """
    if not path.is_file():
        print(f"[WARN] Missing metrics file: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to read {path}: {exc}")
        return None

    results = data.get("results")
    if not isinstance(results, list):
        print(f"[WARN] No 'results' list in {path}")
        return None

    xs: List[float] = []
    ys_top1: List[float] = []
    ys_hv: List[float] = []

    for rec in results:
        if not isinstance(rec, dict):
            continue
        # Prefer generated_num as a proxy for evaluation budget; fall back to Training_step.
        x = rec.get("generated_num", rec.get("Training_step"))
        y1 = rec.get("avg_top1")
        y2 = rec.get("hypervolume", 0.0) # Default to 0.0 if missing (e.g. Greedy)
        if x is None or y1 is None:
            continue
        xs.append(float(x))
        ys_top1.append(float(y1))
        ys_hv.append(float(y2))

    if not xs:
        print(f"[WARN] Empty curve in {path}")
        return None

    # Ensure curves are sorted by x
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys_top1 = [ys_top1[i] for i in order]
    ys_hv = [ys_hv[i] for i in order]

    return xs, ys_top1, ys_hv


def plot_metric_curves(metric: str) -> None:
    """Plot metric ("avg_top1" or "hypervolume") vs generated_num for each seed."""
    assert metric in ("avg_top1", "hypervolume")

    for seed in (42, 43):
        plt.figure(figsize=(6, 4))
        has_any = False
        for algo, seed_map in ALGO_FILES.items():
            path = seed_map.get(seed)
            if path is None:
                continue
            curve = load_curve(path)
            if curve is None:
                continue
            xs, ys_top1, ys_hv = curve
            ys = ys_top1 if metric == "avg_top1" else ys_hv
            plt.plot(xs, ys, label=algo)
            has_any = True

        if not has_any:
            plt.close()
            print(f"[INFO] No data to plot for seed {seed}, metric {metric}.")
            continue

        plt.xlabel("generated_num (evaluations)")
        plt.ylabel(metric)
        plt.title(f"GSCO-Lite {metric} vs evaluations (seed {seed})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PLOTS_DIR / f"gsco_lite_seed{seed}_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved {out_path}")


def collect_final_metrics() -> Dict[int, Dict[str, Dict[str, float]]]:
    """Collect final avg_top1 and hypervolume for each (seed, algo)."""
    out: Dict[int, Dict[str, Dict[str, float]]] = {42: {}, 43: {}}
    for algo, seed_map in ALGO_FILES.items():
        for seed, path in seed_map.items():
            curve = load_curve(path)
            if curve is None:
                continue
            xs, ys_top1, ys_hv = curve
            if not xs:
                continue
            out.setdefault(seed, {})[algo] = {
                "avg_top1": float(ys_top1[-1]),
                "hypervolume": float(ys_hv[-1]),
            }
    return out


def plot_final_bars(metrics: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    """Plot bar charts for final avg_top1 and hypervolume per seed."""
    for seed, alg_map in metrics.items():
        if not alg_map:
            print(f"[INFO] No final metrics for seed {seed}.")
            continue
        algos = sorted(alg_map.keys())
        top1_vals = [alg_map[a]["avg_top1"] for a in algos]
        hv_vals = [alg_map[a]["hypervolume"] for a in algos]

        x = list(range(len(algos)))

        # avg_top1 bar plot
        plt.figure(figsize=(6, 4))
        plt.bar(x, top1_vals, tick_label=algos)
        plt.ylabel("avg_top1 (final)")
        plt.title(f"GSCO-Lite final avg_top1 (seed {seed})")
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.3)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path1 = PLOTS_DIR / f"gsco_lite_seed{seed}_final_avg_top1.png"
        plt.tight_layout()
        plt.savefig(out_path1, dpi=200)
        plt.close()
        print(f"[INFO] Saved {out_path1}")

        # hypervolume bar plot
        plt.figure(figsize=(6, 4))
        plt.bar(x, hv_vals, tick_label=algos)
        plt.ylabel("hypervolume (final)")
        plt.title(f"GSCO-Lite final hypervolume (seed {seed})")
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.3)
        out_path2 = PLOTS_DIR / f"gsco_lite_seed{seed}_final_hypervolume.png"
        plt.tight_layout()
        plt.savefig(out_path2, dpi=200)
        plt.close()
        print(f"[INFO] Saved {out_path2}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Convergence curves
    plot_metric_curves("avg_top1")
    plot_metric_curves("hypervolume")

    # 2) Final-metric bar charts
    final_metrics = collect_final_metrics()
    plot_final_bars(final_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
