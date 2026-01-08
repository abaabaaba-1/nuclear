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
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    label: str
    results_dir: Path
    pattern: str


@dataclass(frozen=True)
class RunChoice:
    seed: int
    runs: dict[str, Path]


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def find_runs(specs: list[AlgoSpec], seeds: list[int], max_eval: int) -> list[RunChoice]:
    out: list[RunChoice] = []
    for seed in seeds:
        runs: dict[str, Path] = {}
        for sp in specs:
            candidates = sorted(sp.results_dir.glob(sp.pattern.format(seed=seed, max_eval=max_eval)))
            chosen = _choose_latest_completed(candidates, max_eval=max_eval)
            if chosen is None:
                raise FileNotFoundError(f"No completed run found for algo={sp.key}, seed={seed} under {sp.results_dir}")
            runs[sp.key] = chosen
        out.append(RunChoice(seed=seed, runs=runs))
    return out


def _plot_multiseed(
    out_png: Path,
    title: str,
    xs: list[float],
    series: dict[str, dict[str, np.ndarray]],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    ax_hv, ax_t1, ax_fr = axes

    x = np.array(xs, dtype=float)

    for k, payload in series.items():
        hv = payload["hv"]
        t1 = payload["top1"]
        fr = payload["feasible_rate"]
        c = colors.get(k, "C0")

        for i in range(hv.shape[0]):
            ax_hv.plot(x, hv[i], color=c, alpha=0.18, linewidth=1.0)
            ax_t1.plot(x, t1[i], color=c, alpha=0.18, linewidth=1.0)
            ax_fr.plot(x, fr[i], color=c, alpha=0.18, linewidth=1.0)

        hv_m, hv_s = hv.mean(axis=0), hv.std(axis=0)
        t1_m, t1_s = t1.mean(axis=0), t1.std(axis=0)
        fr_m, fr_s = fr.mean(axis=0), fr.std(axis=0)

        ax_hv.plot(x, hv_m, color=c, linewidth=2.2, label=f"{labels.get(k, k)} (mean±std)")
        ax_hv.fill_between(x, hv_m - hv_s, hv_m + hv_s, color=c, alpha=0.12)

        ax_t1.plot(x, t1_m, color=c, linewidth=2.2, label=f"{labels.get(k, k)} (mean±std)")
        ax_t1.fill_between(x, t1_m - t1_s, t1_m + t1_s, color=c, alpha=0.12)

        ax_fr.plot(x, fr_m, color=c, linewidth=2.2, label=f"{labels.get(k, k)} (mean±std)")
        ax_fr.fill_between(x, fr_m - fr_s, fr_m + fr_s, color=c, alpha=0.12)

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
    p.add_argument("--max-eval", type=int, default=1000)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    p.add_argument("--out-dir", type=str, default="analysis_outputs/phase5/figures")
    p.add_argument("--skip-missing", action="store_true")
    args = p.parse_args()

    max_eval = int(args.max_eval)
    step = int(args.step)
    seeds = [int(s) for s in args.seeds]
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _seed_tag(ss: list[int]) -> str:
        ss2 = sorted(set(int(x) for x in ss))
        if not ss2:
            return "none"
        if ss2 == list(range(ss2[0], ss2[-1] + 1)):
            return f"{ss2[0]}-{ss2[-1]}"
        return "_".join(str(x) for x in ss2)

    specs = [
        AlgoSpec(
            key="fusionopt_baseline",
            label="FusionOpt Baseline",
            results_dir=_resolve_path("results/stellarator_vmec/fusionopt_v1"),
            pattern="Stellarator_VMEC_Phase4HardV6_Baseline_Budget{max_eval}_seed{seed}_*",
        ),
        AlgoSpec(
            key="fusionopt_llm",
            label="FusionOpt + LLM-SemOp",
            results_dir=_resolve_path("results/stellarator_vmec/fusionopt_v1"),
            pattern="Stellarator_VMEC_Phase4HardV6_LLM_SemOp_Budget{max_eval}_OpenAI_seed{seed}_*",
        ),
        AlgoSpec(
            key="nsga2",
            label="NSGA2",
            results_dir=_resolve_path("results/stellarator_vmec/nsga2"),
            pattern="Stellarator_VMEC_Baseline_NSGA2_Budget{max_eval}_seed{seed}_*",
        ),
        AlgoSpec(
            key="moead",
            label="MOEA/D",
            results_dir=_resolve_path("results/stellarator_vmec/moead"),
            pattern="Stellarator_VMEC_Baseline_MOEAD_Budget{max_eval}_seed{seed}_*",
        ),
    ]

    if args.skip_missing:
        choices: list[RunChoice] = []
        for seed in seeds:
            runs: dict[str, Path] = {}
            ok = True
            for sp in specs:
                candidates = sorted(sp.results_dir.glob(sp.pattern.format(seed=seed, max_eval=max_eval)))
                chosen = _choose_latest_completed(candidates, max_eval=max_eval)
                if chosen is None:
                    ok = False
                    break
                runs[sp.key] = chosen
            if ok:
                choices.append(RunChoice(seed=seed, runs=runs))
    else:
        choices = find_runs(specs=specs, seeds=seeds, max_eval=max_eval)

    if not choices:
        raise SystemExit("No runs found")

    seeds_used = [c.seed for c in choices]

    labels = {sp.key: sp.label for sp in specs}
    colors = {
        "fusionopt_baseline": "C0",
        "fusionopt_llm": "C1",
        "nsga2": "C2",
        "moead": "C3",
    }

    xs_ref: Optional[list[float]] = None
    series_rows: dict[str, dict[str, list[np.ndarray]]] = {
        sp.key: {"hv": [], "top1": [], "feasible_rate": []} for sp in specs
    }

    per_seed: dict[str, dict] = {}
    final_metrics: dict[str, dict[str, list[float]]] = {
        sp.key: {"hv": [], "top1": [], "feasible_rate": []} for sp in specs
    }

    for ch in choices:
        per_seed[str(ch.seed)] = {}
        for sp in specs:
            run_dir = ch.runs[sp.key]
            xs, hv, t1, fr = compute_series(run_dir=run_dir, max_eval=max_eval, step=step)

            if xs_ref is None:
                xs_ref = list(xs)
            if xs_ref != list(xs):
                raise ValueError("Inconsistent x-steps across runs; use a common step/max-eval")

            series_rows[sp.key]["hv"].append(np.array(hv, dtype=float))
            series_rows[sp.key]["top1"].append(np.array(t1, dtype=float))
            series_rows[sp.key]["feasible_rate"].append(np.array(fr, dtype=float))

            per_seed[str(ch.seed)][sp.key] = str(run_dir)

            final_metrics[sp.key]["hv"].append(float(hv[-1]) if hv else 0.0)
            final_metrics[sp.key]["top1"].append(float(t1[-1]) if t1 else 0.0)
            final_metrics[sp.key]["feasible_rate"].append(float(fr[-1]) if fr else 0.0)

    if xs_ref is None:
        raise SystemExit("No runs found")

    series_np: dict[str, dict[str, np.ndarray]] = {}
    for sp in specs:
        series_np[sp.key] = {
            "hv": np.stack(series_rows[sp.key]["hv"], axis=0),
            "top1": np.stack(series_rows[sp.key]["top1"], axis=0),
            "feasible_rate": np.stack(series_rows[sp.key]["feasible_rate"], axis=0),
        }

    seed_tag = _seed_tag(seeds_used)
    out_png = out_dir / f"vmec_fusionopt_vs_baselines_budget{max_eval}_seeds{seed_tag}.png"
    out_json = out_dir / f"vmec_fusionopt_vs_baselines_budget{max_eval}_seeds{seed_tag}.json"

    title = f"VMEC (budget={max_eval}, seeds={','.join(map(str, seeds_used))})"
    _plot_multiseed(
        out_png=out_png,
        title=title,
        xs=xs_ref,
        series=series_np,
        labels=labels,
        colors=colors,
    )

    summary: dict[str, dict] = {}
    for sp in specs:
        hv = np.array(final_metrics[sp.key]["hv"], dtype=float)
        t1 = np.array(final_metrics[sp.key]["top1"], dtype=float)
        fr = np.array(final_metrics[sp.key]["feasible_rate"], dtype=float)
        summary[sp.key] = {
            "label": labels[sp.key],
            "hv": {"per_seed": hv.tolist(), "mean": float(hv.mean()), "std": float(hv.std())},
            "top1": {"per_seed": t1.tolist(), "mean": float(t1.mean()), "std": float(t1.std())},
            "feasible_rate": {"per_seed": fr.tolist(), "mean": float(fr.mean()), "std": float(fr.std())},
        }

    payload = {
        "problem_id": "stellarator_vmec",
        "max_eval": max_eval,
        "step": step,
        "seeds": seeds_used,
        "xs": xs_ref,
        "per_seed": per_seed,
        "labels": labels,
        "series": {
            k: {
                "hv": series_np[k]["hv"].tolist(),
                "top1": series_np[k]["top1"].tolist(),
                "feasible_rate": series_np[k]["feasible_rate"].tolist(),
            }
            for k in series_np
        },
        "final": summary,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(str(out_png))
    print(str(out_json))


if __name__ == "__main__":
    main()
