import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


def _load_goals(run_dir: Path) -> list[str]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    with cfg_path.open("r") as f:
        data = yaml.safe_load(f)
    goals = data.get("goals") if isinstance(data, dict) else None
    if not isinstance(goals, list):
        return []
    return [str(g) for g in goals]


def _load_eval_df(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "evaluations.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_ref_occupancy(run_dir: Path) -> dict:
    path = run_dir / "ref_occupancy.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    out: dict = {
        "ref_occupancy_rows": int(len(df)),
    }
    if df.empty:
        return out

    for c in [
        "occupancy_nonzero_frac",
        "occupancy_zero_frac",
        "occupancy_min",
        "occupancy_max",
        "occupancy_mean",
        "occupancy_std",
    ]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        if s.empty:
            continue
        out[f"{c}_last"] = float(s.iloc[-1])
        out[f"{c}_mean"] = float(s.mean())
    return out


def _maybe_true_pareto_front(cfg: dict, n_obj: int) -> np.ndarray | None:
    bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bench, dict):
        return None
    name = str(bench.get("name", "")).lower().strip()
    if name == "dtlz2":
        try:
            from pymoo.problems.many.dtlz import DTLZ2
        except Exception:
            return None
        try:
            n_var = int(bench.get("n_var", 12))
        except Exception:
            n_var = 12
        try:
            n_pf = int(bench.get("pareto_n_points", 1000))
        except Exception:
            n_pf = 1000

        if n_pf <= 0:
            n_pf = 1000

        try:
            ref_dirs = get_reference_directions("energy", int(n_obj), n_points=int(n_pf))
        except Exception:
            ref_dirs = None
        if ref_dirs is None:
            return None
        try:
            prob = DTLZ2(n_var=int(n_var), n_obj=int(n_obj))
            pf = prob.pareto_front(ref_dirs=ref_dirs)
            return np.asarray(pf, dtype=float)
        except Exception:
            return None
    return None


def _resolve_run_dirs(path: Path) -> list[Path]:
    if (path / "evaluations.csv").exists():
        return [path]

    runs: list[Path] = []
    try:
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "evaluations.csv").exists() and (child / "run_meta.json").exists():
                runs.append(child)
    except Exception:
        runs = []
    if runs:
        return runs

    # Fallback: shallow walk (depth <= 3) for nested result directories.
    base_depth = len(path.parts)
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        if len(root_path.parts) - base_depth > 3:
            continue
        if "evaluations.csv" in files and "run_meta.json" in files:
            runs.append(root_path)
    runs = sorted(list(dict.fromkeys(runs)))
    return runs


def _aggregate_runs(runs: list[dict]) -> dict:
    if not runs:
        return {}
    agg: dict = {
        "n_runs": int(len(runs)),
    }

    for key in ["hv", "igd", "spacing", "n_nd", "n_feasible", "n_rows"]:
        vals = []
        for r in runs:
            v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[f"{key}_mean"] = float(np.mean(arr))
        agg[f"{key}_std"] = float(np.std(arr))

    for key in [
        "occupancy_nonzero_frac_last",
        "occupancy_nonzero_frac_mean",
        "occupancy_zero_frac_last",
        "occupancy_zero_frac_mean",
    ]:
        vals = []
        for r in runs:
            v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[f"{key}_mean"] = float(np.mean(arr))
        agg[f"{key}_std"] = float(np.std(arr))

    # propagate common metadata
    agg["n_obj"] = int(runs[0].get("n_obj", 0) or 0)
    agg["goals"] = runs[0].get("goals")
    agg["algo_id"] = runs[0].get("algo_id")
    return agg


def _objective_cols(df: pd.DataFrame, n_obj: int) -> list[str]:
    cols = []
    for i in range(n_obj):
        c = f"f{i+1}_min"
        if c not in df.columns:
            raise KeyError(f"Missing objective column: {c}")
        cols.append(c)
    return cols


def _metrics_for_run(run_dir: Path) -> dict:
    goals = _load_goals(run_dir)
    df = _load_eval_df(run_dir)
    cfg = _load_config(run_dir)

    n_obj = len(goals)
    if n_obj <= 0:
        i = 1
        while f"f{i}_min" in df.columns:
            i += 1
        n_obj = i - 1
        goals = [f"f{i+1}" for i in range(n_obj)]

    obj_cols = _objective_cols(df, n_obj)

    df2 = df.copy()
    df2["feasible"] = pd.to_numeric(df2.get("feasible", 0), errors="coerce").fillna(0).astype(int)
    feasible_df = df2[df2["feasible"] == 1].copy()

    for c in obj_cols:
        feasible_df[c] = pd.to_numeric(feasible_df[c], errors="coerce")

    feasible_df = feasible_df.dropna(subset=obj_cols)

    F = feasible_df[obj_cols].to_numpy(dtype=float) if not feasible_df.empty else np.zeros((0, n_obj), dtype=float)

    if F.shape[0] > 0:
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F_nd = F[nd_idx]
    else:
        F_nd = np.zeros((0, n_obj), dtype=float)

    ref = np.array([1.1] * n_obj, dtype=float)
    hv = float(HV(ref_point=ref)(F_nd)) if F_nd.shape[0] > 0 else 0.0

    pf = _maybe_true_pareto_front(cfg, n_obj)
    igd = None
    if pf is not None and pf.size and F_nd.shape[0] > 0:
        try:
            igd = float(IGD(pf)(F_nd))
        except Exception:
            igd = None

    spacing = None
    if F_nd.shape[0] >= 2:
        try:
            spacing = float(SpacingIndicator()(F_nd))
        except Exception:
            spacing = None

    best = {}
    if F.shape[0] > 0:
        mins = np.min(F, axis=0)
        for g, v in zip(goals, mins.tolist()):
            best[str(g)] = float(v)

    algo_id = None
    try:
        algo_id = str(df2.get("algo_id").iloc[0]) if "algo_id" in df2.columns and len(df2) > 0 else None
    except Exception:
        algo_id = None

    return {
        "run_dir": str(run_dir),
        "algo_id": algo_id,
        "n_obj": int(n_obj),
        "goals": goals,
        "n_rows": int(len(df2)),
        "n_feasible": int(len(feasible_df)),
        "n_nd": int(F_nd.shape[0]),
        "hv": hv,
        "igd": igd,
        "spacing": spacing,
        "best_min_by_obj": best,
        "F_nd": F_nd.tolist(),
        "F_feasible": F.tolist(),
        **_load_ref_occupancy(run_dir),
    }


def _pair_indices(n_obj: int) -> list[tuple[int, int]]:
    out = []
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            out.append((i, j))
    return out


def _plot_compare(a: dict, b: dict, out_png: Path):
    import matplotlib.pyplot as plt

    n_obj = int(a.get("n_obj", 0) or 0)
    if n_obj <= 1:
        return

    pairs = _pair_indices(n_obj)
    n_plots = len(pairs)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    Fa = np.asarray(a.get("F_feasible") or [], dtype=float)
    Fb = np.asarray(b.get("F_feasible") or [], dtype=float)
    Fna = np.asarray(a.get("F_nd") or [], dtype=float)
    Fnb = np.asarray(b.get("F_nd") or [], dtype=float)

    name_a = a.get("algo_id") or "run_a"
    name_b = b.get("algo_id") or "run_b"

    for idx, (i, j) in enumerate(pairs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        if Fa.size:
            ax.scatter(Fa[:, i], Fa[:, j], s=20, alpha=0.25, label=f"{name_a} feasible")
        if Fb.size:
            ax.scatter(Fb[:, i], Fb[:, j], s=20, alpha=0.25, label=f"{name_b} feasible")
        if Fna.size:
            ax.scatter(Fna[:, i], Fna[:, j], s=70, marker="x", label=f"{name_a} ND")
        if Fnb.size:
            ax.scatter(Fnb[:, i], Fnb[:, j], s=70, marker="x", label=f"{name_b} ND")

        ax.set_xlabel(a.get("goals", [])[i] if a.get("goals") else f"f{i+1}")
        ax.set_ylabel(a.get("goals", [])[j] if a.get("goals") else f"f{j+1}")
        ax.grid(True, alpha=0.3)

    for k in range(n_plots, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_png", default=None)
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    run_dirs_a = _resolve_run_dirs(run_a)
    run_dirs_b = _resolve_run_dirs(run_b)

    runs_a = [_metrics_for_run(p) for p in run_dirs_a]
    runs_b = [_metrics_for_run(p) for p in run_dirs_b]

    comp = {
        "run_a": _aggregate_runs(runs_a) if len(runs_a) > 1 else runs_a[0],
        "run_b": _aggregate_runs(runs_b) if len(runs_b) > 1 else runs_b[0],
        "runs_a": runs_a if len(runs_a) > 1 else None,
        "runs_b": runs_b if len(runs_b) > 1 else None,
    }

    a_ref = comp["run_a"]
    b_ref = comp["run_b"]
    delta = {
        "hv": None,
        "igd": None,
        "spacing": None,
        "n_feasible": None,
        "n_nd": None,
    }
    try:
        if "hv" in a_ref and "hv" in b_ref:
            delta["hv"] = float(b_ref["hv"] - a_ref["hv"])
    except Exception:
        pass
    try:
        if "igd" in a_ref and "igd" in b_ref and a_ref.get("igd") is not None and b_ref.get("igd") is not None:
            delta["igd"] = float(b_ref["igd"] - a_ref["igd"])
    except Exception:
        pass
    try:
        if "spacing" in a_ref and "spacing" in b_ref and a_ref.get("spacing") is not None and b_ref.get("spacing") is not None:
            delta["spacing"] = float(b_ref["spacing"] - a_ref["spacing"])
    except Exception:
        pass
    try:
        if "n_feasible" in a_ref and "n_feasible" in b_ref:
            delta["n_feasible"] = int(b_ref["n_feasible"] - a_ref["n_feasible"])
    except Exception:
        pass
    try:
        if "n_nd" in a_ref and "n_nd" in b_ref:
            delta["n_nd"] = int(b_ref["n_nd"] - a_ref["n_nd"])
    except Exception:
        pass
    comp["delta"] = delta

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2)

    if args.out_png:
        out_png = Path(args.out_png)
        if len(runs_a) == 1 and len(runs_b) == 1:
            _plot_compare(runs_a[0], runs_b[0], out_png)

    print(json.dumps(comp["delta"], indent=2))


if __name__ == "__main__":
    main()
