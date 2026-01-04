import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _infer_seed(path: Path) -> Optional[int]:
    m = re.search(r"seed(\d+)", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON root in {path}")
    return obj


def _extract_curves(obj: Dict[str, Any], side: str) -> Tuple[List[float], List[float], List[float]]:
    curves = obj.get("curves")
    if not isinstance(curves, dict):
        raise KeyError("Missing curves")
    s = curves.get(side)
    if not isinstance(s, dict):
        raise KeyError(f"Missing curves.{side}")
    x = s.get("x")
    fr = s.get("feasible_rate")
    hv = s.get("hv")
    if not (isinstance(x, list) and isinstance(fr, list) and isinstance(hv, list)):
        raise ValueError(f"Invalid curves.{side} fields")
    return [float(v) for v in x], [float(v) for v in fr], [float(v) for v in hv]


def _extract_final_stats(obj: Dict[str, Any], side: str) -> Tuple[Optional[float], Optional[float]]:
    s = obj.get(side)
    if not isinstance(s, dict):
        return None, None
    fr = s.get("feasible_rate")
    if fr is not None:
        try:
            fr = float(fr)
        except Exception:
            fr = None
    curves = obj.get("curves")
    if isinstance(curves, dict) and isinstance(curves.get(side), dict):
        hv = curves[side].get("hv")
        if isinstance(hv, list) and hv:
            try:
                return fr, float(hv[-1])
            except Exception:
                return fr, None
    return fr, None


def _stack(values: List[List[float]]) -> np.ndarray:
    if not values:
        return np.zeros((0, 0), dtype=float)
    n = len(values[0])
    for v in values:
        if len(v) != n:
            raise ValueError("Inconsistent curve lengths")
    return np.array(values, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--label-a", default="heu_on")
    ap.add_argument("--label-b", default="heu_off")
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    objs = []
    seeds = []
    for p in in_paths:
        obj = _load_json(p)
        objs.append(obj)
        sd = _infer_seed(p)
        seeds.append(sd)

    x_ref: Optional[List[float]] = None
    a_fr_curves: List[List[float]] = []
    a_hv_curves: List[List[float]] = []
    b_fr_curves: List[List[float]] = []
    b_hv_curves: List[List[float]] = []

    per_seed: Dict[str, Dict[str, Any]] = {"A": {}, "B": {}}

    for p, obj, sd in zip(in_paths, objs, seeds):
        x_a, fr_a, hv_a = _extract_curves(obj, "A")
        x_b, fr_b, hv_b = _extract_curves(obj, "B")
        if x_a != x_b:
            raise ValueError(f"Mismatched x in {p}")
        if x_ref is None:
            x_ref = x_a
        elif x_a != x_ref:
            raise ValueError(f"Inconsistent x across inputs; expected {x_ref} got {x_a} in {p}")

        a_fr_curves.append(fr_a)
        a_hv_curves.append(hv_a)
        b_fr_curves.append(fr_b)
        b_hv_curves.append(hv_b)

        fr_a_final, hv_a_final = _extract_final_stats(obj, "A")
        fr_b_final, hv_b_final = _extract_final_stats(obj, "B")

        key = str(sd) if sd is not None else p.name
        per_seed["A"][key] = {
            "input": str(p),
            "seed": sd,
            "final_feasible_rate": fr_a_final,
            "final_hv": hv_a_final,
        }
        per_seed["B"][key] = {
            "input": str(p),
            "seed": sd,
            "final_feasible_rate": fr_b_final,
            "final_hv": hv_b_final,
        }

    if x_ref is None:
        raise ValueError("No inputs")

    A_fr = _stack(a_fr_curves)
    A_hv = _stack(a_hv_curves)
    B_fr = _stack(b_fr_curves)
    B_hv = _stack(b_hv_curves)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": [str(p) for p in in_paths],
        "seeds": [s for s in seeds if s is not None],
        "x": [float(v) for v in x_ref],
        "labels": {"A": str(args.label_a), "B": str(args.label_b)},
        "A": {
            "feasible_rate_mean": A_fr.mean(axis=0).tolist(),
            "feasible_rate_std": A_fr.std(axis=0).tolist(),
            "hv_mean": A_hv.mean(axis=0).tolist(),
            "hv_std": A_hv.std(axis=0).tolist(),
            "final_feasible_rate_mean": float(A_fr[:, -1].mean()),
            "final_feasible_rate_std": float(A_fr[:, -1].std()),
            "final_hv_mean": float(A_hv[:, -1].mean()),
            "final_hv_std": float(A_hv[:, -1].std()),
            "per_seed": per_seed["A"],
        },
        "B": {
            "feasible_rate_mean": B_fr.mean(axis=0).tolist(),
            "feasible_rate_std": B_fr.std(axis=0).tolist(),
            "hv_mean": B_hv.mean(axis=0).tolist(),
            "hv_std": B_hv.std(axis=0).tolist(),
            "final_feasible_rate_mean": float(B_fr[:, -1].mean()),
            "final_feasible_rate_std": float(B_fr[:, -1].std()),
            "final_hv_mean": float(B_hv[:, -1].mean()),
            "final_hv_std": float(B_hv[:, -1].std()),
            "per_seed": per_seed["B"],
        },
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_png = out_prefix.with_suffix(".png")

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    x = np.array(x_ref, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    ax0 = axes[0]
    ax0.plot(x, summary["A"]["feasible_rate_mean"], label=summary["labels"]["A"])
    ax0.fill_between(
        x,
        np.array(summary["A"]["feasible_rate_mean"]) - np.array(summary["A"]["feasible_rate_std"]),
        np.array(summary["A"]["feasible_rate_mean"]) + np.array(summary["A"]["feasible_rate_std"]),
        alpha=0.2,
    )
    ax0.plot(x, summary["B"]["feasible_rate_mean"], label=summary["labels"]["B"])
    ax0.fill_between(
        x,
        np.array(summary["B"]["feasible_rate_mean"]) - np.array(summary["B"]["feasible_rate_std"]),
        np.array(summary["B"]["feasible_rate_mean"]) + np.array(summary["B"]["feasible_rate_std"]),
        alpha=0.2,
    )
    ax0.set_xlabel("eval")
    ax0.set_ylabel("feasible_rate")
    ax0.set_ylim(0.0, 1.05)
    ax0.grid(True, alpha=0.25)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(x, summary["A"]["hv_mean"], label=summary["labels"]["A"])
    ax1.fill_between(
        x,
        np.array(summary["A"]["hv_mean"]) - np.array(summary["A"]["hv_std"]),
        np.array(summary["A"]["hv_mean"]) + np.array(summary["A"]["hv_std"]),
        alpha=0.2,
    )
    ax1.plot(x, summary["B"]["hv_mean"], label=summary["labels"]["B"])
    ax1.fill_between(
        x,
        np.array(summary["B"]["hv_mean"]) - np.array(summary["B"]["hv_std"]),
        np.array(summary["B"]["hv_mean"]) + np.array(summary["B"]["hv_std"]),
        alpha=0.2,
    )
    ax1.set_xlabel("eval")
    ax1.set_ylabel("hv")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    print(f"Wrote: {out_json.resolve()}")
    print(f"Wrote: {out_png.resolve()}")


if __name__ == "__main__":
    main()
