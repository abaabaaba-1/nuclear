#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _infer_n_obj(df: pd.DataFrame, goals: list[str]) -> Tuple[int, list[str]]:
    if goals:
        return len(goals), goals

    i = 1
    while f"f{i}_min" in df.columns:
        i += 1
    n_obj = i - 1
    if n_obj <= 0:
        return 0, []
    return n_obj, [f"f{i+1}" for i in range(n_obj)]


def _infer_obj_cols(df: pd.DataFrame, n_obj: int) -> list[str]:
    cols: list[str] = []
    for i in range(n_obj):
        c = f"f{i+1}_min"
        if c not in df.columns:
            raise KeyError(f"Missing objective column: {c}")
        cols.append(c)
    return cols


def _load_eval_df(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "evaluations.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    if df.empty:
        return df

    df2 = df.copy()
    df2["eval_id"] = pd.to_numeric(df2.get("eval_id", 0), errors="coerce").fillna(0).astype(int)
    df2["feasible"] = pd.to_numeric(df2.get("feasible", 0), errors="coerce").fillna(0).astype(int)
    df2["cv"] = pd.to_numeric(df2.get("cv", 0.0), errors="coerce").fillna(0.0).astype(float)
    df2["status"] = df2.get("status", "").astype(str)

    return df2.sort_values("eval_id", kind="stable").reset_index(drop=True)


def _best_so_far(vals: List[float]) -> List[float]:
    out: List[float] = []
    best = float("-inf")
    for v in vals:
        if v > best:
            best = float(v)
        out.append(float(best))
    return out


def _compute_curves(
    df: pd.DataFrame,
    obj_cols: list[str],
    n_obj: int,
    max_eval: Optional[int],
    step: int,
) -> Dict[str, list[float]]:
    if df.empty:
        return {"x": [], "feasible_rate": [], "hv": []}

    max_seen = int(df["eval_id"].max())
    if max_eval is not None:
        max_seen = min(max_seen, int(max_eval))

    if max_seen <= 0:
        return {"x": [], "feasible_rate": [], "hv": []}

    step = int(step) if step is not None else 1
    if step <= 0:
        step = 1

    steps = list(range(step, max_seen + 1, step))
    if steps and steps[-1] != max_seen:
        steps.append(max_seen)

    ref = np.array([1.1] * n_obj, dtype=float)
    hv_ind = HV(ref_point=ref)

    xs: list[float] = []
    feas_rates: list[float] = []
    hvs: list[float] = []

    for s in steps:
        sub = df[df["eval_id"] <= int(s)]
        if sub.empty:
            continue

        denom = int(len(sub))
        feas = sub[(sub["feasible"] == 1) & (sub["status"] == "ok")]
        numer = int(len(feas))
        rate = float(numer) / float(denom) if denom > 0 else 0.0

        feas2 = feas.copy()
        for c in obj_cols:
            feas2[c] = pd.to_numeric(feas2.get(c), errors="coerce")
        feas2 = feas2.dropna(subset=obj_cols)

        if not feas2.empty:
            F = feas2[obj_cols].to_numpy(dtype=float)
            nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
            F_nd = F[nd_idx] if len(nd_idx) else np.zeros((0, n_obj), dtype=float)
            hv_val = float(hv_ind(F_nd)) if F_nd.shape[0] > 0 else 0.0
        else:
            hv_val = 0.0

        xs.append(float(s))
        feas_rates.append(float(rate))
        hvs.append(float(hv_val))

    return {"x": xs, "feasible_rate": feas_rates, "hv": _best_so_far(hvs)}


def _boundary_hit_stats(df: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if df.empty:
        return out
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        v = v[np.isfinite(v)]
        if v.empty:
            continue
        out[c] = {
            "n": int(v.shape[0]),
            "frac_at_0": float((v <= eps).mean()),
            "frac_at_1": float((v >= (1.0 - eps)).mean()),
        }
    return out


def _summary(run_dir: Path, df: pd.DataFrame, goals: list[str], obj_cols: list[str], max_eval: Optional[int]) -> dict:
    df2 = df
    if max_eval is not None and not df2.empty:
        df2 = df2[df2["eval_id"] <= int(max_eval)].copy()

    status_counts = {}
    try:
        status_counts = df2["status"].value_counts(dropna=False).to_dict()
        status_counts = {str(k): int(v) for k, v in status_counts.items()}
    except Exception:
        status_counts = {}

    n_rows = int(len(df2))
    feasible_mask = (df2.get("feasible", 0) == 1) & (df2.get("status", "") == "ok")
    n_feasible = int(feasible_mask.sum()) if n_rows > 0 else 0
    feasible_rate = float(n_feasible) / float(n_rows) if n_rows > 0 else 0.0

    cv = pd.to_numeric(df2.get("cv", 0.0), errors="coerce").fillna(0.0)
    cv_stats = {
        "mean": float(cv.mean()) if n_rows > 0 else 0.0,
        "p50": float(cv.quantile(0.50)) if n_rows > 0 else 0.0,
        "p90": float(cv.quantile(0.90)) if n_rows > 0 else 0.0,
        "p99": float(cv.quantile(0.99)) if n_rows > 0 else 0.0,
        "max": float(cv.max()) if n_rows > 0 else 0.0,
    }

    # Prefer goal_raw columns if present; fall back to f{i}_raw.
    raw_cols: list[str] = []
    if goals:
        for g in goals:
            c = f"{g}_raw"
            if c in df2.columns:
                raw_cols.append(c)
    if not raw_cols:
        for i in range(1, len(obj_cols) + 1):
            c = f"f{i}_raw"
            if c in df2.columns:
                raw_cols.append(c)

    raw_stats = {}
    for c in raw_cols:
        v = pd.to_numeric(df2.get(c), errors="coerce")
        v = v[np.isfinite(v)]
        if v.empty:
            continue
        raw_stats[c] = {
            "mean": float(v.mean()),
            "p10": float(v.quantile(0.10)),
            "p50": float(v.quantile(0.50)),
            "p90": float(v.quantile(0.90)),
            "min": float(v.min()),
            "max": float(v.max()),
            "frac_eq_0": float((v == 0.0).mean()),
        }

    min_boundary = _boundary_hit_stats(df2[feasible_mask].copy(), obj_cols)

    return {
        "run_dir": str(run_dir),
        "n_rows": n_rows,
        "n_feasible": n_feasible,
        "feasible_rate": feasible_rate,
        "status_counts": status_counts,
        "cv_stats": cv_stats,
        "raw_stats": raw_stats,
        "feasible_min_score_boundary_hits": min_boundary,
    }


def _plot(
    out_png: Path,
    title: str,
    a_label: str,
    b_label: str,
    a_curves: Dict[str, list[float]],
    b_curves: Dict[str, list[float]],
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    max_eval: Optional[int],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax0, ax1, ax2 = axes

    # Feasible rate curve
    ax0.plot(a_curves["x"], a_curves["feasible_rate"], marker="o", markersize=2.5, linewidth=1.2, label=a_label)
    ax0.plot(b_curves["x"], b_curves["feasible_rate"], marker="o", markersize=2.5, linewidth=1.2, label=b_label)
    ax0.set_xlabel("Eval")
    ax0.set_ylabel("Cumulative feasible rate")
    ax0.set_title("Feasibility")
    ax0.set_ylim(0.0, 1.0)
    ax0.legend(fontsize=8)

    # HV curve
    ax1.plot(a_curves["x"], a_curves["hv"], marker="o", markersize=2.5, linewidth=1.2, label=a_label)
    ax1.plot(b_curves["x"], b_curves["hv"], marker="o", markersize=2.5, linewidth=1.2, label=b_label)
    ax1.set_xlabel("Eval")
    ax1.set_ylabel("HV (best-so-far, feasible & ok)")
    ax1.set_title("Hypervolume")
    ax1.legend(fontsize=8)

    # CV histogram
    bins = 30
    a_cv = pd.to_numeric(a_df.get("cv", 0.0), errors="coerce").fillna(0.0)
    b_cv = pd.to_numeric(b_df.get("cv", 0.0), errors="coerce").fillna(0.0)
    if max_eval is not None:
        a_cv = a_cv[a_df["eval_id"] <= int(max_eval)]
        b_cv = b_cv[b_df["eval_id"] <= int(max_eval)]

    ax2.hist(a_cv, bins=bins, alpha=0.55, label=a_label)
    ax2.hist(b_cv, bins=bins, alpha=0.55, label=b_label)
    ax2.set_xlabel("cv")
    ax2.set_ylabel("count")
    ax2.set_title("Constraint violation (cv)")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir_a", type=str, help="Protocol run dir A (e.g., repair_on)")
    parser.add_argument("run_dir_b", type=str, help="Protocol run dir B (e.g., repair_off)")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--out-prefix", type=str, default=None)
    args = parser.parse_args()

    run_a = Path(args.run_dir_a)
    if not run_a.is_absolute():
        run_a = PROJECT_ROOT / run_a
    run_b = Path(args.run_dir_b)
    if not run_b.is_absolute():
        run_b = PROJECT_ROOT / run_b

    df_a = _load_eval_df(run_a)
    df_b = _load_eval_df(run_b)

    goals = _load_goals(run_a)
    n_obj, goals2 = _infer_n_obj(df_a, goals)
    if n_obj <= 0:
        raise SystemExit("Cannot infer number of objectives from evaluations.csv")

    obj_cols = _infer_obj_cols(df_a, n_obj)

    max_eval = args.max_eval
    if max_eval is None:
        try:
            max_eval = int(min(df_a["eval_id"].max(), df_b["eval_id"].max()))
        except Exception:
            max_eval = None

    curves_a = _compute_curves(df_a, obj_cols=obj_cols, n_obj=n_obj, max_eval=max_eval, step=args.step)
    curves_b = _compute_curves(df_b, obj_cols=obj_cols, n_obj=n_obj, max_eval=max_eval, step=args.step)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_obj": int(n_obj),
        "goals": goals2,
        "obj_cols": obj_cols,
        "max_eval": int(max_eval) if max_eval is not None else None,
        "curves": {
            "A": curves_a,
            "B": curves_b,
        },
        "A": _summary(run_a, df_a, goals2, obj_cols, max_eval=max_eval),
        "B": _summary(run_b, df_b, goals2, obj_cols, max_eval=max_eval),
    }

    if args.out_prefix is not None:
        out_prefix = Path(args.out_prefix)
        if not out_prefix.is_absolute():
            out_prefix = PROJECT_ROOT / out_prefix
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = PROJECT_ROOT / "analysis_outputs" / "figure"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_prefix = out_dir / f"phase3_heu_repair_ablation_{ts}"

    out_json = Path(str(out_prefix) + ".json")
    out_png = Path(str(out_prefix) + ".png")

    if args.title is not None:
        title = str(args.title)
    else:
        title = f"Phase3 HeuRepairOp Ablation (max_eval={max_eval})"
    _plot(
        out_png=out_png,
        title=title,
        a_label=str(args.label_a),
        b_label=str(args.label_b),
        a_curves=curves_a,
        b_curves=curves_b,
        a_df=df_a,
        b_df=df_b,
        max_eval=max_eval,
    )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_json.resolve()}")
    print(f"Wrote: {out_png.resolve()}")


if __name__ == "__main__":
    main()
