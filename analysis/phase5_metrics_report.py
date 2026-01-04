#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MetricReport:
    run_dir: str
    problem_id: str
    algo_id: str
    seed: Optional[int]
    n_rows: int
    n_ok_feasible: int
    feasible_rate: float
    hv_phase4: float
    top1_total_phase4: float
    n_obj: int
    ref_point: List[float]
    out_of_range_low_rate: Dict[str, float]
    out_of_range_high_rate: Dict[str, float]
    sat0_rate: Dict[str, float]
    sat1_rate: Dict[str, float]
    hv_unclipped: Optional[float]


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(x)


def _non_dominated(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return F
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    if nd_idx is None or len(nd_idx) == 0:
        return np.zeros((0, F.shape[1]), dtype=float)
    return F[np.asarray(nd_idx, dtype=int)]


def _compute_hv(F: np.ndarray, ref: np.ndarray) -> float:
    if F.size == 0:
        return 0.0
    hv_ind = HV(ref_point=ref)
    return float(hv_ind(_non_dominated(F)))


def _infer_goals(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    goals = cfg.get("goals")
    if isinstance(goals, list) and all(isinstance(x, str) for x in goals):
        return list(goals)

    out: List[str] = []
    i = 1
    while f"f{i}_min" in df.columns:
        out.append(f"f{i}")
        i += 1
    return out


def _objective_ranges(cfg: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    ranges = cfg.get("objective_ranges")
    if not isinstance(ranges, dict):
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    for k, v in ranges.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        lo = _safe_float(v[0])
        hi = _safe_float(v[1])
        if lo is None or hi is None:
            continue
        if hi == lo:
            continue
        out[str(k)] = (float(lo), float(hi))
    return out


def _objective_directions(cfg: Dict[str, Any], goals: List[str]) -> Dict[str, str]:
    dirs = cfg.get("optimization_direction")
    if isinstance(dirs, list) and len(dirs) == len(goals):
        out: Dict[str, str] = {}
        for g, d in zip(goals, dirs):
            if isinstance(d, str):
                out[g] = d
        return out
    return {g: "min" for g in goals}


def _compute_unclipped_F(df_ok: pd.DataFrame, goals: List[str], ranges: Dict[str, Tuple[float, float]], directions: Dict[str, str]) -> Optional[np.ndarray]:
    if df_ok.empty:
        return None

    cols: List[str] = []
    for g in goals:
        if f"{g}_raw" in df_ok.columns:
            cols.append(f"{g}_raw")
        else:
            cols.append("")

    raw_mat: List[np.ndarray] = []
    for i, g in enumerate(goals):
        c = cols[i]
        if not c:
            return None
        if g not in ranges:
            return None
        lo, hi = ranges[g]
        denom = float(hi - lo)
        if denom == 0.0:
            return None

        v = pd.to_numeric(df_ok[c], errors="coerce").to_numpy(dtype=float)
        if directions.get(g, "min") == "max":
            f = (hi - v) / denom
        else:
            f = (v - lo) / denom
        raw_mat.append(f.reshape(-1, 1))

    return np.concatenate(raw_mat, axis=1)


def _plot_diagnostics(
    out_png: Path,
    goals: List[str],
    out_low: Dict[str, float],
    out_high: Dict[str, float],
    sat0: Dict[str, float],
    sat1: Dict[str, float],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    x = np.arange(len(goals), dtype=float)
    w = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 3.6))

    ax.bar(x - 1.5 * w, [out_low.get(g, 0.0) for g in goals], width=w, label="raw < lo")
    ax.bar(x - 0.5 * w, [out_high.get(g, 0.0) for g in goals], width=w, label="raw > hi")
    ax.bar(x + 0.5 * w, [sat0.get(g, 0.0) for g in goals], width=w, label="f*_min ~ 0")
    ax.bar(x + 1.5 * w, [sat1.get(g, 0.0) for g in goals], width=w, label="f*_min ~ 1")

    ax.set_xticks(x)
    ax.set_xticklabels(goals)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Clipping / saturation diagnostics")
    ax.legend(fontsize=8, ncols=4)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_report(run_dir: Path, ref_point: Optional[List[float]] = None, eps: float = 1e-9) -> MetricReport:
    cfg = _load_yaml(run_dir / "config.yaml")
    meta = _load_yaml(run_dir / "run_meta.json")

    df = pd.read_csv(run_dir / "evaluations.csv")
    df = df.copy()
    df["feasible"] = pd.to_numeric(df.get("feasible", 0), errors="coerce").fillna(0).astype(int)
    df["status"] = df.get("status", "").astype(str)

    goals = _infer_goals(df, cfg)
    n_obj = len(goals)
    if n_obj <= 0:
        raise ValueError("Cannot infer number of objectives")

    obj_cols = [f"f{i+1}_min" for i in range(n_obj)]
    for c in obj_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    df_ok = df[(df["feasible"] == 1) & (df["status"] == "ok")].dropna(subset=obj_cols).copy()

    n_rows = int(len(df))
    n_ok = int(len(df_ok))
    feasible_rate = float(n_ok) / float(n_rows) if n_rows > 0 else 0.0

    if ref_point is None:
        ref_point = [1.1] * n_obj
    if len(ref_point) != n_obj:
        raise ValueError(f"ref_point length {len(ref_point)} != n_obj {n_obj}")
    ref = np.array(ref_point, dtype=float)

    F = df_ok[obj_cols].to_numpy(dtype=float) if n_ok > 0 else np.zeros((0, n_obj), dtype=float)
    hv_phase4 = _compute_hv(F, ref)

    if "total" in df_ok.columns and n_ok > 0:
        tot = pd.to_numeric(df_ok["total"], errors="coerce")
        top1 = float(tot.max()) if np.isfinite(tot.max()) else 0.0
    else:
        top1 = float((1.0 - np.mean(F, axis=1)).max()) if F.size else 0.0

    ranges = _objective_ranges(cfg)
    directions = _objective_directions(cfg, goals)

    out_low: Dict[str, float] = {}
    out_high: Dict[str, float] = {}
    for g in goals:
        c = f"{g}_raw" if f"{g}_raw" in df_ok.columns else None
        if c is None or g not in ranges or n_ok == 0:
            out_low[g] = 0.0
            out_high[g] = 0.0
            continue
        lo, hi = ranges[g]
        v = pd.to_numeric(df_ok[c], errors="coerce").to_numpy(dtype=float)
        out_low[g] = float(np.mean(v < float(lo)))
        out_high[g] = float(np.mean(v > float(hi)))

    sat0: Dict[str, float] = {}
    sat1: Dict[str, float] = {}
    for i, g in enumerate(goals):
        c = obj_cols[i]
        if n_ok == 0:
            sat0[g] = 0.0
            sat1[g] = 0.0
            continue
        v = df_ok[c].to_numpy(dtype=float)
        sat0[g] = float(np.mean(v <= eps))
        sat1[g] = float(np.mean(v >= (1.0 - eps)))

    F_unclipped = _compute_unclipped_F(df_ok, goals, ranges, directions)
    hv_unclipped = None
    if F_unclipped is not None:
        hv_unclipped = _compute_hv(F_unclipped, ref)

    return MetricReport(
        run_dir=str(run_dir),
        problem_id=str(meta.get("problem_id") or cfg.get("protocol", {}).get("problem_id") or ""),
        algo_id=str(meta.get("algo_id") or cfg.get("protocol", {}).get("algo_id") or ""),
        seed=int(meta.get("seed")) if isinstance(meta.get("seed"), int) else None,
        n_rows=n_rows,
        n_ok_feasible=n_ok,
        feasible_rate=float(feasible_rate),
        hv_phase4=float(hv_phase4),
        top1_total_phase4=float(top1),
        n_obj=int(n_obj),
        ref_point=[float(x) for x in ref_point],
        out_of_range_low_rate=out_low,
        out_of_range_high_rate=out_high,
        sat0_rate=sat0,
        sat1_rate=sat1,
        hv_unclipped=hv_unclipped,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis_outputs/phase5/metrics")
    p.add_argument("--ref", type=float, nargs="*", default=None)
    p.add_argument("--eps", type=float, default=1e-9)
    args = p.parse_args()

    run_dir = _resolve_path(args.run_dir)
    out_dir = _resolve_path(args.out_dir)

    ref_point = list(args.ref) if args.ref is not None and len(args.ref) > 0 else None

    rep = compute_report(run_dir=run_dir, ref_point=ref_point, eps=float(args.eps))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir.name

    out_json = out_dir / f"{stem}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rep.__dict__, f, indent=2, sort_keys=True)

    out_png = out_dir / f"{stem}_diagnostics.png"
    goals = _infer_goals(pd.read_csv(run_dir / "evaluations.csv"), _load_yaml(run_dir / "config.yaml"))
    _plot_diagnostics(
        out_png=out_png,
        goals=goals,
        out_low=rep.out_of_range_low_rate,
        out_high=rep.out_of_range_high_rate,
        sat0=rep.sat0_rate,
        sat1=rep.sat1_rate,
    )

    print(str(out_json))
    print(str(out_png))


if __name__ == "__main__":
    main()
