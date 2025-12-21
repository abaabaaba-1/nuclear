#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Pareto front for VMEC 3-objective optimization results.

This script is tailored for the stellarator VMEC problem with goals:
- volume (maximize)
- aspect_ratio (minimize)
- magnetic_shear (maximize)

It reads a MOLLM checkpoint PKL (same style as ``read_checkpoint.py``),
extracts metrics from Item.property["original_results"], computes the
Pareto front over feasible designs, and saves a PNG with pairwise
projections of the Pareto front.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# Optional: make sure project root is on sys.path so that pickled Items
# referencing algorithm.base can be imported correctly.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from algorithm.base import Item, HistoryBuffer  # type: ignore
except Exception:
    # Fallback minimal stubs so that unpickling still works even if the
    # full project environment is not available.
    class Item:  # type: ignore
        def __init__(self):
            self.value = ""
            self.property: Dict[str, Any] = {}
            self.total: float = 0.0

    class HistoryBuffer:  # type: ignore
        pass


def load_checkpoint(filepath: str) -> Any:
    """Load a PKL checkpoint with a robust fallback encoding."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")

    with path.open("rb") as f:
        try:
            data = pickle.load(f)
        except (UnicodeDecodeError, ModuleNotFoundError):
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    return data


def iter_items_from_data(data: Any) -> List[Any]:
    """Extract candidate Items from a generic MOLLM checkpoint structure.

    This follows the logic used in problem/stellarator_vmec/plot_vmec_checkpoint.py
    but simplified for our purpose.
    """
    items: List[Any] = []

    if isinstance(data, dict):
        for key in [
            "final_pops",
            "all_mols",
            "init_pops",
            "history",
            "results",
            "population",
        ]:
            v = data.get(key)
            if isinstance(v, list) and v:
                for entry in v:
                    if isinstance(entry, (list, tuple)) and entry:
                        cand = entry[0]
                    else:
                        cand = entry
                    if hasattr(cand, "value"):
                        items.append(cand)

    # Fallback: top-level list of Items
    if not items and isinstance(data, list):
        for entry in data:
            if hasattr(entry, "value"):
                items.append(entry)

    # De-duplicate by candidate string
    uniq: List[Any] = []
    seen = set()
    for it in items:
        sig = getattr(it, "value", None)
        if sig and sig not in seen:
            uniq.append(it)
            seen.add(sig)
    return uniq


def parse_vmec_metrics(item: Any) -> Dict[str, Any]:
    """Parse VMEC metrics (volume, aspect_ratio, magnetic_shear) from Item."""
    metrics: Dict[str, Any] = {}
    prop = getattr(item, "property", {}) or {}

    is_feasible = 1.0
    if "original_results" in prop:
        orig = prop.get("original_results", {}) or {}
        cr = prop.get("constraint_results", {}) or {}
        metrics["volume"] = orig.get("volume")
        metrics["aspect_ratio"] = orig.get("aspect_ratio")
        metrics["magnetic_shear"] = orig.get("magnetic_shear")
        is_feasible = cr.get("is_feasible", 1.0)
    else:
        # Older / fallback format
        metrics["volume"] = prop.get("volume")
        metrics["aspect_ratio"] = prop.get("aspect_ratio")
        metrics["magnetic_shear"] = prop.get("magnetic_shear")
        is_feasible = prop.get("is_feasible", 1.0)

    try:
        metrics["is_feasible"] = float(is_feasible) if is_feasible is not None else 1.0
    except Exception:
        metrics["is_feasible"] = 1.0

    if hasattr(item, "total") and getattr(item, "total") is not None:
        try:
            metrics["total"] = float(getattr(item, "total"))
        except Exception:
            pass

    metrics["candidate"] = getattr(item, "value", None)
    return metrics


def build_dataframe(items: List[Any]) -> pd.DataFrame:
    """Build a pandas DataFrame with VMEC metrics for each candidate."""
    rows: List[Dict[str, Any]] = []
    for it in items:
        m = parse_vmec_metrics(it)
        if any(m.get(k) is not None for k in ("volume", "aspect_ratio", "magnetic_shear")):
            rows.append(m)

    if not rows:
        raise RuntimeError("No valid VMEC metrics found in checkpoint.")

    df = pd.DataFrame(rows)
    return df


def identify_pareto_front(df: pd.DataFrame, use_feasible_only: bool = True) -> pd.DataFrame:
    """Mark Pareto-front points in the DataFrame.

    We treat this as a mixed max/min problem:
    - volume: maximize
    - aspect_ratio: minimize
    - magnetic_shear: maximize

    For Pareto comparison, we convert all objectives to a minimization form
    (by negating the maximization objectives).
    """
    df_use = df.copy()
    if use_feasible_only and "is_feasible" in df_use.columns:
        df_use = df_use[df_use["is_feasible"] >= 0.5]

    df_use = df_use.dropna(subset=["volume", "aspect_ratio", "magnetic_shear"])
    if df_use.empty:
        raise RuntimeError("No valid (feasible) points with all three metrics.")

    points = df_use[["volume", "aspect_ratio", "magnetic_shear"]].to_numpy(dtype=float)

    # Convert to minimization objectives
    objs = np.empty_like(points)
    objs[:, 0] = -points[:, 0]  # volume: max -> min by negation
    objs[:, 1] = points[:, 1]   # aspect_ratio: min
    objs[:, 2] = -points[:, 2]  # magnetic_shear: max -> min by negation

    n = objs.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is no worse in all and strictly better in at least one
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                is_pareto[i] = False
                break

    df = df.copy()
    df["is_pareto"] = False
    df.loc[df_use.index, "is_pareto"] = is_pareto
    return df


def plot_pareto(df: pd.DataFrame, out_png: Path) -> None:
    """Plot pairwise Pareto front projections and save to PNG."""
    if "is_feasible" in df.columns:
        df_plot = df[df["is_feasible"] >= 0.5].copy()
    else:
        df_plot = df.copy()

    df_plot = df_plot.dropna(subset=["volume", "aspect_ratio", "magnetic_shear"])
    if df_plot.empty:
        raise RuntimeError("No feasible points to plot.")

    df_front = df_plot[df_plot.get("is_pareto", False)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # volume vs aspect_ratio
    ax = axes[0]
    ax.scatter(df_plot["volume"], df_plot["aspect_ratio"], s=15, alpha=0.3, label="Feasible")
    if not df_front.empty:
        ax.scatter(df_front["volume"], df_front["aspect_ratio"], s=40, c="red", label="Pareto front")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Aspect ratio")
    ax.set_title("Volume vs Aspect ratio")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # volume vs magnetic_shear
    ax = axes[1]
    ax.scatter(df_plot["volume"], df_plot["magnetic_shear"], s=15, alpha=0.3, label="Feasible")
    if not df_front.empty:
        ax.scatter(df_front["volume"], df_front["magnetic_shear"], s=40, c="red", label="Pareto front")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Magnetic shear")
    ax.set_title("Volume vs Magnetic shear")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # aspect_ratio vs magnetic_shear
    ax = axes[2]
    ax.scatter(df_plot["aspect_ratio"], df_plot["magnetic_shear"], s=15, alpha=0.3, label="Feasible")
    if not df_front.empty:
        ax.scatter(df_front["aspect_ratio"], df_front["magnetic_shear"], s=40, c="red", label="Pareto front")
    ax.set_xlabel("Aspect ratio")
    ax.set_ylabel("Magnetic shear")
    ax.set_title("Aspect ratio vs Magnetic shear")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("VMEC 3-objective Pareto front (volume, aspect_ratio, magnetic_shear)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Pareto front figure saved to: {out_png.resolve()}")


def plot_pareto_3d(df: pd.DataFrame, out_png: Path) -> None:
    if "is_feasible" in df.columns:
        df_plot = df[df["is_feasible"] >= 0.5].copy()
    else:
        df_plot = df.copy()

    df_plot = df_plot.dropna(subset=["volume", "aspect_ratio", "magnetic_shear"])
    if df_plot.empty:
        raise RuntimeError("No feasible points to plot.")

    df_front = df_plot[df_plot.get("is_pareto", False)].copy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 背景可行点：做子采样，弱化显示
    if not df_front.empty:
        bg = df_plot.drop(index=df_front.index)
    else:
        bg = df_plot.copy()
    if len(bg) > 400:
        bg = bg.sample(400, random_state=0)

    if not bg.empty:
        ax.scatter(
            bg["volume"],
            bg["aspect_ratio"],
            bg["magnetic_shear"],
            s=10,
            alpha=0.15,
            color="#8cb3d9",
            label="Feasible (sampled)",
        )

    # 使用帕累托前沿点三角剖分绘制前沿曲面
    if not df_front.empty and len(df_front) >= 3:
        x = df_front["volume"].to_numpy(dtype=float)
        y = df_front["aspect_ratio"].to_numpy(dtype=float)
        z = df_front["magnetic_shear"].to_numpy(dtype=float)
        tri = Triangulation(x, y)
        surf = ax.plot_trisurf(
            tri,
            z,
            cmap="viridis",
            alpha=0.85,
            linewidth=0.2,
            edgecolor="none",
        )

        # 用边缘点再强调一次帕累托前沿散点
        ax.scatter(
            x,
            y,
            z,
            s=25,
            color="red",
            edgecolor="k",
            linewidth=0.3,
            label="Pareto front",
        )
    else:
        # 没有足够的帕累托点时，退化为普通散点图
        ax.scatter(
            df_plot["volume"],
            df_plot["aspect_ratio"],
            df_plot["magnetic_shear"],
            s=20,
            alpha=0.5,
            color="red",
            label="Feasible",
        )

    ax.set_xlabel("Volume")
    ax.set_ylabel("Aspect ratio")
    ax.set_zlabel("Magnetic shear")
    ax.set_title("VMEC Pareto front (3D)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    # 调整视角和长宽比，让曲面更容易观察
    ax.view_init(elev=25, azim=-55)
    try:
        ax.set_box_aspect((1.6, 1.0, 0.6))
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"3D Pareto front figure saved to: {out_png.resolve()}")


def main() -> None:
    import argparse

    default_pkl = (
        "moo_results/zgca,gemini-2.5-flash-nothinking/mols/"
        "volume_aspect_ratio_magnetic_shear_stellarator_vmec_3_obj_42.pkl"
    )

    parser = argparse.ArgumentParser(
        description="Plot Pareto front for stellarator VMEC 3-objective optimization results",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=default_pkl,
        help="path to checkpoint PKL file (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output PNG path (default: <pkl_basename>_pareto.png)",
    )
    parser.add_argument(
        "--include_infeasible",
        action="store_true",
        help="include infeasible points when computing Pareto front",
    )

    args = parser.parse_args()

    data = load_checkpoint(args.pkl)
    items = iter_items_from_data(data)
    if not items:
        raise RuntimeError("Failed to extract any candidate Item from PKL.")

    print(f"Extracted {len(items)} candidate items from {args.pkl}")
    df = build_dataframe(items)
    print(f"DataFrame shape before filtering: {df.shape}")

    df = identify_pareto_front(df, use_feasible_only=not args.include_infeasible)
    num_front = int(df.get("is_pareto", pd.Series(dtype=bool)).sum())
    print(f"Identified {num_front} Pareto-front points")

    if args.out is not None:
        out_path_2d = Path(args.out)
        out_path_3d = out_path_2d.with_name(out_path_2d.stem + "_3d.png")
    else:
        base = Path(args.pkl)
        out_path_2d = base.with_name(base.stem + "_pareto.png")
        out_path_3d = base.with_name(base.stem + "_pareto_3d.png")

    plot_pareto(df, out_path_2d)
    plot_pareto_3d(df, out_path_3d)


if __name__ == "__main__":
    main()
