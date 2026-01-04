#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunChoice:
    seed: int
    baseline_dir: Path
    llm_dir: Path


def _best_so_far(vals: list[float]) -> list[float]:
    if not vals:
        return []
    out: list[float] = []
    best = float("-inf")
    for v in vals:
        if v > best:
            best = v
        out.append(float(best))
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
    cols: list[str] = []
    for i in range(n_obj):
        c = f"f{i+1}_min"
        if c not in df.columns:
            raise KeyError(f"Missing objective column: {c}")
        cols.append(c)
    return cols


def _infer_n_obj_from_df(df: pd.DataFrame) -> int:
    i = 1
    while f"f{i}_min" in df.columns:
        i += 1
    return i - 1


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
    df2["status"] = df2.get("status", "").astype(str)
    return df2.sort_values("eval_id", kind="stable").reset_index(drop=True)


def compute_series(run_dir: Path, max_eval: int, step: int) -> tuple[list[float], list[float], list[float]]:
    df = _load_eval_df(run_dir)
    if df.empty:
        return [], [], []

    goals = _load_goals(run_dir)
    n_obj = len(goals)
    if n_obj <= 0:
        n_obj = _infer_n_obj_from_df(df)
    if n_obj <= 0:
        raise ValueError(f"Cannot infer number of objectives from {run_dir / 'evaluations.csv'}")

    obj_cols = _infer_obj_cols(df, n_obj)
    for c in obj_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    df_ok = df[(df["feasible"] == 1) & (df["status"] == "ok")].dropna(subset=obj_cols).copy()

    steps = list(range(step, max_eval + 1, step))
    if not steps or steps[-1] != max_eval:
        steps.append(max_eval)

    ref = np.array([1.1] * n_obj, dtype=float)
    hv_ind = HV(ref_point=ref)

    xs: list[float] = []
    hvs: list[float] = []
    top1s: list[float] = []

    for s in steps:
        sub = df_ok[df_ok["eval_id"] <= int(s)]
        if sub.empty:
            xs.append(float(s))
            hvs.append(0.0)
            top1s.append(0.0)
            continue

        F = sub[obj_cols].to_numpy(dtype=float)
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F_nd = F[nd_idx] if len(nd_idx) else np.zeros((0, n_obj), dtype=float)
        hv_val = float(hv_ind(F_nd)) if F_nd.shape[0] > 0 else 0.0

        if "total" in sub.columns:
            tot = pd.to_numeric(sub["total"], errors="coerce")
            top1 = float(tot.max()) if np.isfinite(tot.max()) else 0.0
        else:
            top1 = float((1.0 - np.mean(F, axis=1)).max())

        xs.append(float(s))
        hvs.append(float(hv_val))
        top1s.append(float(top1))

    return xs, _best_so_far(hvs), top1s


def _run_timestamp(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            with meta.open("r", encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp")
            if isinstance(ts, str) and ts:
                return ts
        except Exception:
            pass
    return run_dir.name


def _choose_latest_completed(candidates: Iterable[Path], max_eval: int) -> Optional[Path]:
    best: Optional[Path] = None
    best_key: Optional[str] = None
    for p in candidates:
        eval_path = p / "evaluations.csv"
        if not eval_path.exists():
            continue
        try:
            df = pd.read_csv(eval_path, usecols=["eval_id"])
            mx = int(pd.to_numeric(df["eval_id"], errors="coerce").fillna(0).max())
        except Exception:
            continue
        if mx < int(max_eval):
            continue

        key = _run_timestamp(p)
        if best is None or (best_key is not None and key > best_key) or best_key is None:
            best = p
            best_key = key
    return best


def _plot_pair(out_png: Path, title: str, baseline: tuple[list[float], list[float], list[float]], llm: tuple[list[float], list[float], list[float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.24, 3.84))
    ax_hv, ax_top1 = axes

    bx, bhv, bt1 = baseline
    lx, lhv, lt1 = llm

    ax_hv.plot(bx, bhv, marker="o", markersize=2.5, linewidth=1.2, label="Baseline")
    ax_hv.plot(lx, lhv, marker="o", markersize=2.5, linewidth=1.2, label="LLM-SemOp")

    ax_top1.plot(bx, bt1, marker="o", markersize=2.5, linewidth=1.2, label="Baseline")
    ax_top1.plot(lx, lt1, marker="o", markersize=2.5, linewidth=1.2, label="LLM-SemOp")

    ax_hv.set_xlabel("Generated candidates")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV convergence")

    ax_top1.set_xlabel("Generated candidates")
    ax_top1.set_ylabel("avg_top1")
    ax_top1.set_title("Top-1 convergence (avg_top1)")

    ax_hv.legend(fontsize=8)
    ax_top1.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_multiseed_summary(
    out_png: Path,
    title: str,
    xs: list[float],
    baseline_hv: np.ndarray,
    llm_hv: np.ndarray,
    baseline_t1: np.ndarray,
    llm_t1: np.ndarray,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.24, 3.84))
    ax_hv, ax_top1 = axes

    x = np.array(xs, dtype=float)

    for i in range(baseline_hv.shape[0]):
        ax_hv.plot(x, baseline_hv[i], color="C0", alpha=0.25, linewidth=1.0)
        ax_hv.plot(x, llm_hv[i], color="C1", alpha=0.25, linewidth=1.0)
        ax_top1.plot(x, baseline_t1[i], color="C0", alpha=0.25, linewidth=1.0)
        ax_top1.plot(x, llm_t1[i], color="C1", alpha=0.25, linewidth=1.0)

    b_hv_mean = baseline_hv.mean(axis=0)
    b_hv_std = baseline_hv.std(axis=0)
    l_hv_mean = llm_hv.mean(axis=0)
    l_hv_std = llm_hv.std(axis=0)

    b_t1_mean = baseline_t1.mean(axis=0)
    b_t1_std = baseline_t1.std(axis=0)
    l_t1_mean = llm_t1.mean(axis=0)
    l_t1_std = llm_t1.std(axis=0)

    ax_hv.plot(x, b_hv_mean, color="C0", linewidth=2.2, label="Baseline (mean±std)")
    ax_hv.fill_between(x, b_hv_mean - b_hv_std, b_hv_mean + b_hv_std, color="C0", alpha=0.15)
    ax_hv.plot(x, l_hv_mean, color="C1", linewidth=2.2, label="LLM-SemOp (mean±std)")
    ax_hv.fill_between(x, l_hv_mean - l_hv_std, l_hv_mean + l_hv_std, color="C1", alpha=0.15)

    ax_top1.plot(x, b_t1_mean, color="C0", linewidth=2.2, label="Baseline (mean±std)")
    ax_top1.fill_between(x, b_t1_mean - b_t1_std, b_t1_mean + b_t1_std, color="C0", alpha=0.15)
    ax_top1.plot(x, l_t1_mean, color="C1", linewidth=2.2, label="LLM-SemOp (mean±std)")
    ax_top1.fill_between(x, l_t1_mean - l_t1_std, l_t1_mean + l_t1_std, color="C1", alpha=0.15)

    ax_hv.set_xlabel("Generated candidates")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV convergence")
    ax_hv.legend(fontsize=8)

    ax_top1.set_xlabel("Generated candidates")
    ax_top1.set_ylabel("avg_top1")
    ax_top1.set_title("Top-1 convergence (avg_top1)")
    ax_top1.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_runs(results_dir: Path, seeds: list[int], max_eval: int) -> list[RunChoice]:
    out: list[RunChoice] = []

    for seed in seeds:
        baseline_candidates = sorted(results_dir.glob(f"Stellarator_VMEC_Phase4HardV6_Baseline_Budget{max_eval}_seed{seed}_*"))
        llm_candidates = sorted(results_dir.glob(f"Stellarator_VMEC_Phase4HardV6_LLM_SemOp_Budget{max_eval}_OpenAI_seed{seed}_*"))

        baseline_dir = _choose_latest_completed(baseline_candidates, max_eval=max_eval)
        llm_dir = _choose_latest_completed(llm_candidates, max_eval=max_eval)

        if baseline_dir is None:
            raise FileNotFoundError(f"No completed baseline run found for seed={seed} under {results_dir}")
        if llm_dir is None:
            raise FileNotFoundError(f"No completed LLM-SemOp run found for seed={seed} under {results_dir}")

        out.append(RunChoice(seed=seed, baseline_dir=baseline_dir, llm_dir=llm_dir))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/stellarator_vmec/fusionopt_v1")
    parser.add_argument("--out-dir", type=str, default="showcase/figures")
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46, 47])
    args = parser.parse_args()

    results_dir = _resolve_path(args.results_dir)
    out_dir = _resolve_path(args.out_dir)

    max_eval = int(args.max_eval)
    step = int(args.step)
    seeds = [int(s) for s in args.seeds]

    choices = find_runs(results_dir=results_dir, seeds=seeds, max_eval=max_eval)

    xs_ref: Optional[list[float]] = None
    baseline_hv_rows: list[np.ndarray] = []
    llm_hv_rows: list[np.ndarray] = []
    baseline_t1_rows: list[np.ndarray] = []
    llm_t1_rows: list[np.ndarray] = []

    for ch in choices:
        baseline_series = compute_series(ch.baseline_dir, max_eval=max_eval, step=step)
        llm_series = compute_series(ch.llm_dir, max_eval=max_eval, step=step)

        if xs_ref is None:
            xs_ref = list(baseline_series[0])
        if xs_ref != list(baseline_series[0]) or xs_ref != list(llm_series[0]):
            raise ValueError("Inconsistent x-steps across runs; please use a common step/max-eval")

        out_png = out_dir / f"phase4_hardv6_baseline_vs_llm_semop_budget{max_eval}_seed{ch.seed}.png"
        title = f"Phase4HardV6 (budget={max_eval}, seed={ch.seed})"
        _plot_pair(out_png=out_png, title=title, baseline=baseline_series, llm=llm_series)

        baseline_hv_rows.append(np.array(baseline_series[1], dtype=float))
        llm_hv_rows.append(np.array(llm_series[1], dtype=float))
        baseline_t1_rows.append(np.array(baseline_series[2], dtype=float))
        llm_t1_rows.append(np.array(llm_series[2], dtype=float))

    if xs_ref is None:
        raise SystemExit("No runs found")

    baseline_hv = np.stack(baseline_hv_rows, axis=0)
    llm_hv = np.stack(llm_hv_rows, axis=0)
    baseline_t1 = np.stack(baseline_t1_rows, axis=0)
    llm_t1 = np.stack(llm_t1_rows, axis=0)

    out_png = out_dir / f"phase4_hardv6_baseline_vs_llm_semop_budget{max_eval}_multiseed.png"
    title = f"Phase4HardV6 (budget={max_eval}, seeds={min(seeds)}–{max(seeds)})"
    _plot_multiseed_summary(
        out_png=out_png,
        title=title,
        xs=xs_ref,
        baseline_hv=baseline_hv,
        llm_hv=llm_hv,
        baseline_t1=baseline_t1,
        llm_t1=llm_t1,
    )

    print(f"Wrote per-seed figures + multiseed summary under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
