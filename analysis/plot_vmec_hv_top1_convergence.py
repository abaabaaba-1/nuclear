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
"results/figure/vmec_3obj_hv_top1_convergence.png".
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _best_so_far(vals: List[float]) -> List[float]:
    if not vals:
        return []
    out: List[float] = []
    best = float("-inf")
    for v in vals:
        if v > best:
            best = v
        out.append(best)
    return out


def _load_goals(run_dir: Path) -> list[str]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    try:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return []
    if not isinstance(cfg, dict):
        return []
    goals = cfg.get("goals")
    if not isinstance(goals, list):
        return []
    return [str(g) for g in goals]


def _infer_obj_cols(df: pd.DataFrame, n_obj: int) -> list[str]:
    cols = []
    for i in range(n_obj):
        c = f"f{i+1}_min"
        if c not in df.columns:
            raise KeyError(f"Missing objective column: {c}")
        cols.append(c)
    return cols


def _compute_series_from_protocol(
    run_dir: Path,
    max_eval: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    eval_path = run_dir / "evaluations.csv"
    if not eval_path.exists():
        raise FileNotFoundError(str(eval_path))

    goals = _load_goals(run_dir)
    df = pd.read_csv(eval_path)
    n_obj = len(goals)
    if n_obj <= 0:
        i = 1
        while f"f{i}_min" in df.columns:
            i += 1
        n_obj = i - 1
        goals = [f"f{i+1}" for i in range(n_obj)]
    if n_obj <= 0:
        raise ValueError(f"Cannot infer number of objectives from {eval_path}")

    obj_cols = _infer_obj_cols(df, n_obj)

    df2 = df.copy()
    df2["eval_id"] = pd.to_numeric(df2.get("eval_id", 0), errors="coerce").fillna(0).astype(int)
    df2["feasible"] = pd.to_numeric(df2.get("feasible", 0), errors="coerce").fillna(0).astype(int)
    df2["status"] = df2.get("status", "").astype(str)
    for c in obj_cols:
        df2[c] = pd.to_numeric(df2.get(c), errors="coerce")

    feasible_df = df2[(df2["feasible"] == 1) & (df2["status"] == "ok")].dropna(subset=obj_cols).copy()
    if feasible_df.empty:
        return [], [], []

    # Logging frequency: prefer config.optimization.log_freq; fallback to 50.
    log_freq = 50
    try:
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            with cfg_path.open("r") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                opt = cfg.get("optimization") or {}
                if isinstance(opt, dict):
                    log_freq = int(opt.get("log_freq", log_freq))
    except Exception:
        log_freq = 50
    if log_freq <= 0:
        log_freq = 50

    actual_max_eval = int(df2["eval_id"].max())
    if max_eval is not None:
        try:
            max_eval = int(max_eval)
        except Exception:
            max_eval = None
    if max_eval is not None and max_eval > 0:
        actual_max_eval = min(actual_max_eval, int(max_eval))

    steps = list(range(log_freq, actual_max_eval + 1, log_freq))
    if steps and steps[-1] != actual_max_eval:
        steps.append(actual_max_eval)
    if not steps:
        steps = [actual_max_eval]

    xs: List[float] = []
    hvs: List[float] = []
    top1s: List[float] = []
    ref = np.array([1.1] * n_obj, dtype=float)
    hv_ind = HV(ref_point=ref)

    for s in steps:
        sub = feasible_df[feasible_df["eval_id"] <= int(s)]
        if sub.empty:
            continue

        F = sub[obj_cols].to_numpy(dtype=float)
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F_nd = F[nd_idx] if len(nd_idx) else np.zeros((0, n_obj), dtype=float)
        hv_val = float(hv_ind(F_nd)) if F_nd.shape[0] > 0 else 0.0

        # Define top1 consistently:
        # - if 'total' column exists, use max(total)
        # - else infer from RewardingSystem convention: total = 1 - mean(f*_min)
        if "total" in sub.columns:
            tot = pd.to_numeric(sub["total"], errors="coerce")
            top1 = float(tot.max()) if np.isfinite(tot.max()) else 0.0
        else:
            top1 = float((1.0 - np.mean(F, axis=1)).max())

        xs.append(float(s))
        hvs.append(float(hv_val))
        top1s.append(float(top1))

    return xs, _best_so_far(hvs), top1s


def _compute_series_from_legacy_json(path: Path) -> Tuple[List[float], List[float], List[float]]:
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
            x = idx

        xs.append(float(x))
        hvs.append(float(hv))
        top1s.append(float(top1))

    if not xs:
        raise ValueError(f"No valid hv/top1 records found in {path}")

    return xs, _best_so_far(hvs), top1s


def load_results(
    path: Path,
    max_eval: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Load either a protocol run dir (evaluations.csv) or a legacy JSON log."""
    if path.is_dir():
        return _compute_series_from_protocol(path, max_eval=max_eval)
    xs, hvs, top1s = _compute_series_from_legacy_json(path)
    if max_eval is None:
        return xs, hvs, top1s

    try:
        max_eval_int = int(max_eval)
    except Exception:
        return xs, hvs, top1s
    if max_eval_int <= 0:
        return xs, hvs, top1s

    xs2: List[float] = []
    hvs2: List[float] = []
    top1s2: List[float] = []
    for x, hv, t1 in zip(xs, hvs, top1s):
        if float(x) <= float(max_eval_int):
            xs2.append(float(x))
            hvs2.append(float(hv))
            top1s2.append(float(t1))
        else:
            break
    return xs2, hvs2, top1s2


def pretty_label(path: Path) -> str:
    """Generate a short label from the input path."""
    if path.is_dir():
        try:
            meta = path / "run_meta.json"
            if meta.exists():
                with meta.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                algo_id = data.get("algo_id")
                if algo_id:
                    return str(algo_id)
        except Exception:
            pass
        return path.name

    name = path.stem
    if "baseline_GA" in name:
        return "Baseline GA"
    if "baseline_NSGA2" in name:
        return "Baseline NSGA-II"
    if "baseline_SMSEMOA" in name:
        return "Baseline SMSEMOA"
    return name


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
        help="JSON result files OR protocol run directories to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Truncate curves to this evaluation budget (x-axis).",
    )
    parser.add_argument(
        "--pad-to-max-eval",
        action="store_true",
        help="If set and --max-eval is provided, extend shorter curves as a flat line to max-eval.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: results/figure/vmec_3obj_hv_top1_convergence.png)",
    )

    args = parser.parse_args()

    paths: List[Path] = []
    for name in args.json_files:
        p = Path(name)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        try:
            xs, hvs, top1s = load_results(p, max_eval=args.max_eval)
        except Exception as exc:
            print(f"[WARN] Failed to load {p}: {exc}")
            continue

        if args.pad_to_max_eval and args.max_eval is not None:
            try:
                max_eval_int = int(args.max_eval)
            except Exception:
                max_eval_int = None
            if max_eval_int is not None and max_eval_int > 0 and xs and xs[-1] < float(max_eval_int):
                xs = list(xs) + [float(max_eval_int)]
                hvs = list(hvs) + [float(hvs[-1])]
                top1s = list(top1s) + [float(top1s[-1])]

        paths.append((p, xs, hvs, top1s))

    if not paths:
        raise SystemExit("No valid JSON files to plot.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_hv, ax_top1 = axes

    for path, xs, hvs, top1s in paths:
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
        out_dir = PROJECT_ROOT / "results" / "figure"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vmec_3obj_hv_top1_convergence.png"

    fig.savefig(out_path, dpi=150)
    print(f"Convergence figure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
