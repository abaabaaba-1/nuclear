#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
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
    a_dir: Path
    b_dir: Path


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


def _infer_n_obj_from_df(df: pd.DataFrame) -> int:
    i = 1
    while f"f{i}_min" in df.columns:
        i += 1
    return i - 1


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
    df2["status"] = df2.get("status", "").astype(str)
    return df2.sort_values("eval_id", kind="stable").reset_index(drop=True)


def compute_series(run_dir: Path, max_eval: int, step: int) -> tuple[list[float], list[float], list[float], list[float]]:
    df = _load_eval_df(run_dir)
    if df.empty:
        return [], [], [], []

    goals = _load_goals(run_dir)
    n_obj = len(goals)
    if n_obj <= 0:
        n_obj = _infer_n_obj_from_df(df)
    if n_obj <= 0:
        raise ValueError(f"Cannot infer number of objectives from {run_dir / 'evaluations.csv'}")

    obj_cols = _infer_obj_cols(df, n_obj)
    for c in obj_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    steps = list(range(step, max_eval + 1, step))
    if not steps or steps[-1] != max_eval:
        steps.append(max_eval)

    ref = np.array([1.1] * n_obj, dtype=float)
    hv_ind = HV(ref_point=ref)

    xs: list[float] = []
    hvs: list[float] = []
    top1s: list[float] = []
    frs: list[float] = []

    for s in steps:
        sub_all = df[df["eval_id"] <= int(s)]
        if sub_all.empty:
            xs.append(float(s))
            hvs.append(0.0)
            top1s.append(0.0)
            frs.append(0.0)
            continue

        sub_ok = sub_all[(sub_all["feasible"] == 1) & (sub_all["status"] == "ok")].dropna(subset=obj_cols)
        fr = float(len(sub_ok)) / float(len(sub_all)) if len(sub_all) > 0 else 0.0

        if sub_ok.empty:
            xs.append(float(s))
            hvs.append(0.0)
            top1s.append(0.0)
            frs.append(float(fr))
            continue

        F = sub_ok[obj_cols].to_numpy(dtype=float)
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F_nd = F[np.asarray(nd_idx, dtype=int)] if nd_idx is not None and len(nd_idx) else np.zeros((0, n_obj), dtype=float)
        hv_val = float(hv_ind(F_nd)) if F_nd.shape[0] > 0 else 0.0

        if "total" in sub_ok.columns:
            tot = pd.to_numeric(sub_ok["total"], errors="coerce")
            top1 = float(tot.max()) if np.isfinite(tot.max()) else 0.0
        else:
            top1 = float((1.0 - np.mean(F, axis=1)).max())

        xs.append(float(s))
        hvs.append(float(hv_val))
        top1s.append(float(top1))
        frs.append(float(fr))

    return xs, _best_so_far(hvs), top1s, frs


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


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_runs(
    a_results_dir: Path,
    b_results_dir: Path,
    a_pattern: str,
    b_pattern: str,
    seeds: list[int],
    max_eval: int,
    skip_missing: bool,
) -> list[RunChoice]:
    out: list[RunChoice] = []

    for seed in seeds:
        a_candidates = sorted(a_results_dir.glob(a_pattern.format(seed=seed, max_eval=max_eval)))
        b_candidates = sorted(b_results_dir.glob(b_pattern.format(seed=seed, max_eval=max_eval)))

        a_dir = _choose_latest_completed(a_candidates, max_eval=max_eval)
        b_dir = _choose_latest_completed(b_candidates, max_eval=max_eval)

        if a_dir is None or b_dir is None:
            if skip_missing:
                continue
            if a_dir is None:
                raise FileNotFoundError(f"No completed run found for A seed={seed} under {a_results_dir}")
            raise FileNotFoundError(f"No completed run found for B seed={seed} under {b_results_dir}")

        out.append(RunChoice(seed=seed, a_dir=a_dir, b_dir=b_dir))

    return out


def _plot_multiseed(
    out_png: Path,
    title: str,
    xs: list[float],
    a_hv: np.ndarray,
    b_hv: np.ndarray,
    a_t1: np.ndarray,
    b_t1: np.ndarray,
    a_fr: np.ndarray,
    b_fr: np.ndarray,
    label_a: str,
    label_b: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    ax_hv, ax_t1, ax_fr = axes

    x = np.array(xs, dtype=float)

    for i in range(a_hv.shape[0]):
        ax_hv.plot(x, a_hv[i], color="C0", alpha=0.25, linewidth=1.0)
        ax_hv.plot(x, b_hv[i], color="C1", alpha=0.25, linewidth=1.0)
        ax_t1.plot(x, a_t1[i], color="C0", alpha=0.25, linewidth=1.0)
        ax_t1.plot(x, b_t1[i], color="C1", alpha=0.25, linewidth=1.0)
        ax_fr.plot(x, a_fr[i], color="C0", alpha=0.25, linewidth=1.0)
        ax_fr.plot(x, b_fr[i], color="C1", alpha=0.25, linewidth=1.0)

    def mean_std(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return a.mean(axis=0), a.std(axis=0)

    a_hv_m, a_hv_s = mean_std(a_hv)
    b_hv_m, b_hv_s = mean_std(b_hv)
    a_t1_m, a_t1_s = mean_std(a_t1)
    b_t1_m, b_t1_s = mean_std(b_t1)
    a_fr_m, a_fr_s = mean_std(a_fr)
    b_fr_m, b_fr_s = mean_std(b_fr)

    ax_hv.plot(x, a_hv_m, color="C0", linewidth=2.2, label=f"{label_a} (mean±std)")
    ax_hv.fill_between(x, a_hv_m - a_hv_s, a_hv_m + a_hv_s, color="C0", alpha=0.15)
    ax_hv.plot(x, b_hv_m, color="C1", linewidth=2.2, label=f"{label_b} (mean±std)")
    ax_hv.fill_between(x, b_hv_m - b_hv_s, b_hv_m + b_hv_s, color="C1", alpha=0.15)

    ax_t1.plot(x, a_t1_m, color="C0", linewidth=2.2, label=f"{label_a} (mean±std)")
    ax_t1.fill_between(x, a_t1_m - a_t1_s, a_t1_m + a_t1_s, color="C0", alpha=0.15)
    ax_t1.plot(x, b_t1_m, color="C1", linewidth=2.2, label=f"{label_b} (mean±std)")
    ax_t1.fill_between(x, b_t1_m - b_t1_s, b_t1_m + b_t1_s, color="C1", alpha=0.15)

    ax_fr.plot(x, a_fr_m, color="C0", linewidth=2.2, label=f"{label_a} (mean±std)")
    ax_fr.fill_between(x, a_fr_m - a_fr_s, a_fr_m + a_fr_s, color="C0", alpha=0.15)
    ax_fr.plot(x, b_fr_m, color="C1", linewidth=2.2, label=f"{label_b} (mean±std)")
    ax_fr.fill_between(x, b_fr_m - b_fr_s, b_fr_m + b_fr_s, color="C1", alpha=0.15)

    ax_hv.set_xlabel("Evaluations")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV (Phase4 metric)")
    ax_hv.legend(fontsize=8)

    ax_t1.set_xlabel("Evaluations")
    ax_t1.set_ylabel("top1(total)")
    ax_t1.set_title("Top-1 (total)")
    ax_t1.legend(fontsize=8)

    ax_fr.set_xlabel("Evaluations")
    ax_fr.set_ylabel("feasible_rate")
    ax_fr.set_title("Feasible rate")
    ax_fr.set_ylim(0.0, 1.0)
    ax_fr.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a-results-dir", type=str, default="results/stellarator_vmec/nsga2")
    p.add_argument("--b-results-dir", type=str, default="results/stellarator_vmec/moead")
    p.add_argument("--a-label", type=str, default="NSGA2")
    p.add_argument("--b-label", type=str, default="MOEA/D")
    p.add_argument(
        "--a-pattern",
        type=str,
        default="Stellarator_VMEC_Baseline_NSGA2_Budget{max_eval}_seed{seed}_*",
    )
    p.add_argument(
        "--b-pattern",
        type=str,
        default="Stellarator_VMEC_Baseline_MOEAD_Budget{max_eval}_seed{seed}_*",
    )
    p.add_argument("--out-dir", type=str, default="analysis_outputs/phase5/figures")
    p.add_argument("--max-eval", type=int, default=1000)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    p.add_argument("--skip-missing", action="store_true")
    args = p.parse_args()

    def _slug(s: str) -> str:
        out = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
        return out or "run"

    a_results_dir = _resolve_path(args.a_results_dir)
    b_results_dir = _resolve_path(args.b_results_dir)
    out_dir = _resolve_path(args.out_dir)

    max_eval = int(args.max_eval)
    step = int(args.step)
    seeds = [int(s) for s in args.seeds]

    choices = find_runs(
        a_results_dir=a_results_dir,
        b_results_dir=b_results_dir,
        a_pattern=str(args.a_pattern),
        b_pattern=str(args.b_pattern),
        seeds=seeds,
        max_eval=max_eval,
        skip_missing=bool(args.skip_missing),
    )

    if not choices:
        raise SystemExit("No runs found")

    seeds_used = [c.seed for c in choices]

    def _seed_tag(ss: list[int]) -> str:
        ss2 = sorted(set(int(x) for x in ss))
        if not ss2:
            return "none"
        if ss2 == list(range(ss2[0], ss2[-1] + 1)):
            return f"{ss2[0]}-{ss2[-1]}"
        return "_".join(str(x) for x in ss2)

    xs_ref: Optional[list[float]] = None
    a_hv_rows: list[np.ndarray] = []
    b_hv_rows: list[np.ndarray] = []
    a_t1_rows: list[np.ndarray] = []
    b_t1_rows: list[np.ndarray] = []
    a_fr_rows: list[np.ndarray] = []
    b_fr_rows: list[np.ndarray] = []

    per_seed: dict = {}
    for ch in choices:
        a = compute_series(ch.a_dir, max_eval=max_eval, step=step)
        b = compute_series(ch.b_dir, max_eval=max_eval, step=step)

        if xs_ref is None:
            xs_ref = list(a[0])
        if xs_ref != list(a[0]) or xs_ref != list(b[0]):
            raise ValueError("Inconsistent x-steps across runs; use a common step/max-eval")

        a_hv_rows.append(np.array(a[1], dtype=float))
        b_hv_rows.append(np.array(b[1], dtype=float))
        a_t1_rows.append(np.array(a[2], dtype=float))
        b_t1_rows.append(np.array(b[2], dtype=float))
        a_fr_rows.append(np.array(a[3], dtype=float))
        b_fr_rows.append(np.array(b[3], dtype=float))

        per_seed[str(ch.seed)] = {
            "a_dir": str(ch.a_dir),
            "b_dir": str(ch.b_dir),
        }

    if xs_ref is None:
        raise SystemExit("No runs found")

    a_hv = np.stack(a_hv_rows, axis=0)
    b_hv = np.stack(b_hv_rows, axis=0)
    a_t1 = np.stack(a_t1_rows, axis=0)
    b_t1 = np.stack(b_t1_rows, axis=0)
    a_fr = np.stack(a_fr_rows, axis=0)
    b_fr = np.stack(b_fr_rows, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)

    a_slug = _slug(args.a_label)
    b_slug = _slug(args.b_label)

    seed_tag = _seed_tag(seeds_used)
    out_png = out_dir / f"vmec_baselines_{a_slug}_vs_{b_slug}_budget{max_eval}_seeds{seed_tag}.png"
    title = f"VMEC baselines (budget={max_eval}, seeds={','.join(map(str, seeds_used))})"
    _plot_multiseed(
        out_png=out_png,
        title=title,
        xs=xs_ref,
        a_hv=a_hv,
        b_hv=b_hv,
        a_t1=a_t1,
        b_t1=b_t1,
        a_fr=a_fr,
        b_fr=b_fr,
        label_a=str(args.a_label),
        label_b=str(args.b_label),
    )

    out_json = out_dir / f"vmec_baselines_{a_slug}_vs_{b_slug}_budget{max_eval}_seeds{seed_tag}.json"
    payload = {
        "problem_id": "stellarator_vmec",
        "max_eval": max_eval,
        "step": step,
        "seeds": seeds_used,
        "xs": xs_ref,
        "per_seed": per_seed,
        "a": {
            "label": str(args.a_label),
            "hv": a_hv.tolist(),
            "top1": a_t1.tolist(),
            "feasible_rate": a_fr.tolist(),
        },
        "b": {
            "label": str(args.b_label),
            "hv": b_hv.tolist(),
            "top1": b_t1.tolist(),
            "feasible_rate": b_fr.tolist(),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(str(out_png))
    print(str(out_json))


if __name__ == "__main__":
    main()
