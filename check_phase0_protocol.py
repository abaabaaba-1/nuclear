from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except Exception:
        return default


def _safe_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default


def _find_run_dirs(results_dir: Path) -> list[Path]:
    runs: list[Path] = []
    if not results_dir.exists():
        return runs

    for root, _dirs, files in os.walk(results_dir):
        if "evaluations.csv" in files and "run_meta.json" in files:
            runs.append(Path(root))
    runs.sort()
    return runs


def _load_yaml(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        try:
            with path.open("r") as f:
                return yaml.safe_load(f)
        except Exception:
            return None


def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with path.open("r") as f:
                return json.load(f)
        except Exception:
            return None


def _summarize_run(run_dir: Path) -> dict:
    summary: dict = {
        "run_dir": str(run_dir),
        "ok": True,
        "missing_files": [],
        "missing_generation_files": [],
        "missing_generation_columns": [],
        "missing_columns": [],
        "n_rows": 0,
        "n_feasible": 0,
        "status_counts": {},
        "objectives": {},
    }

    for fname in ("config.yaml", "run_meta.json", "evaluations.csv"):
        if not (run_dir / fname).exists():
            summary["missing_files"].append(fname)
            summary["ok"] = False

    pareto_path = run_dir / "final" / "pareto_front.csv"
    if not pareto_path.exists():
        summary["missing_files"].append("final/pareto_front.csv")
        summary["ok"] = False

    generations_dir = run_dir / "generations"
    gen_files = []
    try:
        if generations_dir.exists() and generations_dir.is_dir():
            gen_files = sorted(generations_dir.glob("gen_*_pop.csv"))
    except Exception:
        gen_files = []

    if not gen_files:
        summary["missing_files"].append("generations/gen_XXXX_pop.csv")
        summary["ok"] = False

    cfg = _load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None
    goals = []
    if isinstance(cfg, dict):
        goals = cfg.get("goals") or []
        if not isinstance(goals, list):
            goals = []

    eval_path = run_dir / "evaluations.csv"
    if not eval_path.exists():
        return summary

    with eval_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        required_cols = [
            "eval_id",
            "run_id",
            "generation",
            "ind_idx",
            "problem_id",
            "algo_id",
            "decision_id",
            "status",
            "sim_message",
            "feasible",
            "cv",
            "eval_tier",
        ]

        # At least one objective must be present (prefer f*_min per protocol).
        if isinstance(cfg, dict) and isinstance(goals, list) and goals:
            for i, g in enumerate(goals):
                # allow either {goal}_min or f{i}_min
                if (f"{g}_min" not in cols) and (f"f{i+1}_min" not in cols):
                    required_cols.append(f"f{i+1}_min")
        missing_cols = [c for c in required_cols if c not in cols]
        if missing_cols:
            summary["missing_columns"] = missing_cols
            summary["ok"] = False

        status_counts: dict[str, int] = {}
        n_rows = 0
        n_feasible = 0
        best_obj: dict[str, float] = {}

        for row in reader:
            n_rows += 1
            status = (row.get("status") or "").strip() or "<empty>"
            status_counts[status] = status_counts.get(status, 0) + 1

            feasible = _safe_int(row.get("feasible", 0), 0)
            if feasible == 1:
                n_feasible += 1

            for i, g in enumerate(goals):
                key = f"{g}_min" if f"{g}_min" in cols else f"f{i+1}_min"
                v = _safe_float(row.get(key))
                if v is None:
                    continue
                if (g not in best_obj) or (v < best_obj[g]):
                    best_obj[g] = v

        summary["n_rows"] = n_rows
        summary["n_feasible"] = n_feasible
        summary["status_counts"] = status_counts
        summary["objectives"] = best_obj

    # Validate generation snapshot schema (minimal required columns).
    if gen_files:
        gen_path = gen_files[0]
        try:
            with gen_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                gen_cols = list(reader.fieldnames or [])
        except Exception:
            gen_cols = []

        gen_required = [
            "generation",
            "ind_idx",
            "decision_id",
            "eval_id",
            "cv",
            "feasible",
        ]
        if isinstance(cfg, dict) and isinstance(goals, list) and goals:
            for i, _g in enumerate(goals):
                gen_required.append(f"f{i+1}_min")

        missing_gen_cols = [c for c in gen_required if c not in gen_cols]
        if missing_gen_cols:
            summary["missing_generation_columns"] = missing_gen_cols
            summary["ok"] = False

    return summary


def _repair_run(run_dir: Path) -> bool:
    repaired_any = False

    eval_path = run_dir / "evaluations.csv"
    if not eval_path.exists():
        return False

    cfg = _load_yaml(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None
    goals = []
    if isinstance(cfg, dict):
        goals = cfg.get("goals") or []
        if not isinstance(goals, list):
            goals = []

    try:
        with eval_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return False

    if not rows:
        return False

    num_obj = 0
    if goals:
        num_obj = len(goals)
    else:
        sample_cols = list(rows[0].keys())
        i = 1
        while f"f{i}_min" in sample_cols:
            i += 1
        num_obj = i - 1

    if num_obj <= 0:
        return False

    # 1) Backfill generation snapshots
    generations_dir = run_dir / "generations"
    gen_files = []
    try:
        if generations_dir.exists() and generations_dir.is_dir():
            gen_files = sorted(generations_dir.glob("gen_*_pop.csv"))
    except Exception:
        gen_files = []

    if not gen_files:
        try:
            generations_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        by_gen: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            g = _safe_int(r.get("generation", 0), 0)
            if g < 0:
                g = 0
            by_gen[g].append(r)

        fieldnames = [
            "generation",
            "ind_idx",
            "parent_ids",
            "rank",
            "crowding",
            "apd",
            "selected_flag",
            "decision_id",
            "eval_id",
            "cv",
            "feasible",
        ]
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_raw")
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_min")

        for g, rs in sorted(by_gen.items(), key=lambda kv: kv[0]):
            out_path = generations_dir / f"gen_{int(g):04d}_pop.csv"
            if out_path.exists():
                continue
            try:
                with out_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rs:
                        out_row = {
                            "generation": int(g),
                            "ind_idx": r.get("ind_idx", ""),
                            "parent_ids": "",
                            "rank": "",
                            "crowding": "",
                            "apd": "",
                            "selected_flag": "",
                            "decision_id": r.get("decision_id", ""),
                            "eval_id": r.get("eval_id", ""),
                            "cv": r.get("cv", ""),
                            "feasible": r.get("feasible", ""),
                        }
                        for i in range(num_obj):
                            out_row[f"f{i+1}_raw"] = r.get(f"f{i+1}_raw", "")
                            out_row[f"f{i+1}_min"] = r.get(f"f{i+1}_min", "")
                        writer.writerow(out_row)
                repaired_any = True
            except Exception:
                continue

    # 2) Backfill final/pareto_front.csv
    pareto_path = run_dir / "final" / "pareto_front.csv"
    if not pareto_path.exists():
        cand_rows = []
        F = []

        for r in rows:
            feasible = _safe_int(r.get("feasible", 0), 0)
            if feasible != 1:
                continue
            fvec = []
            ok = True
            for i in range(num_obj):
                v = _safe_float(r.get(f"f{i+1}_min"), None)
                if v is None:
                    ok = False
                    break
                fvec.append(float(v))
            if not ok:
                continue
            cand_rows.append(r)
            F.append(fvec)

        if cand_rows:
            objs = np.asarray(F, dtype=float)
            n = objs.shape[0]
            is_pareto = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_pareto[i]:
                    continue
                dom = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
                dom[i] = False
                if np.any(dom):
                    is_pareto[i] = False

            out_rows = [cand_rows[i] for i in range(n) if is_pareto[i]]
            if out_rows:
                out_dir = run_dir / "final"
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                fieldnames = [
                    "decision_id",
                    "decision_json",
                    "status",
                    "sim_message",
                    "cv",
                    "feasible",
                    "g1",
                ]
                for g in goals:
                    fieldnames.append(f"{g}_raw")
                for g in goals:
                    fieldnames.append(f"{g}_min")
                for i in range(num_obj):
                    fieldnames.append(f"f{i+1}_raw")
                for i in range(num_obj):
                    fieldnames.append(f"f{i+1}_min")

                try:
                    with pareto_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in out_rows:
                            writer.writerow({k: r.get(k, "") for k in fieldnames})
                    repaired_any = True
                except Exception:
                    pass

    return repaired_any


def _print_summary(summary: dict):
    run_dir = summary.get("run_dir")
    ok = summary.get("ok")
    print(f"\nRUN: {run_dir}")
    print(f"  ok: {ok}")

    missing_files = summary.get("missing_files") or []
    if missing_files:
        print(f"  missing_files: {missing_files}")

    missing_cols = summary.get("missing_columns") or []
    if missing_cols:
        print(f"  missing_columns: {missing_cols}")

    missing_gen_cols = summary.get("missing_generation_columns") or []
    if missing_gen_cols:
        print(f"  missing_generation_columns: {missing_gen_cols}")

    print(f"  n_rows: {summary.get('n_rows')}")
    print(f"  n_feasible: {summary.get('n_feasible')}")

    status_counts = summary.get("status_counts") or {}
    if status_counts:
        keys = sorted(status_counts.keys())
        flat = ", ".join([f"{k}={status_counts[k]}" for k in keys])
        print(f"  status_counts: {flat}")

    obj = summary.get("objectives") or {}
    if obj:
        keys = sorted(obj.keys())
        flat = ", ".join([f"{k}={obj[k]:.6g}" for k in keys])
        print(f"  best_objectives(min-space): {flat}")


def _compare_runs(a: dict, b: dict, tol: float):
    ok = True

    if a.get("n_rows") != b.get("n_rows"):
        ok = False
        print(f"n_rows mismatch: {a.get('n_rows')} vs {b.get('n_rows')}")

    if a.get("n_feasible") != b.get("n_feasible"):
        ok = False
        print(f"n_feasible mismatch: {a.get('n_feasible')} vs {b.get('n_feasible')}")

    ao = a.get("objectives") or {}
    bo = b.get("objectives") or {}
    keys = sorted(set(ao.keys()) | set(bo.keys()))
    for k in keys:
        av = ao.get(k)
        bv = bo.get(k)
        if av is None or bv is None:
            ok = False
            print(f"objective '{k}' missing in one run: {av} vs {bv}")
            continue
        if abs(av - bv) > tol:
            ok = False
            print(f"objective '{k}' mismatch: {av} vs {bv} (tol={tol})")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("RUN_DIR_A", "RUN_DIR_B"))
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--repair", action="store_true", help="Backfill missing Phase 0 artifacts for runs under results_dir")
    args = parser.parse_args()

    if args.compare is not None:
        run_a = Path(args.compare[0]).resolve()
        run_b = Path(args.compare[1]).resolve()
        sa = _summarize_run(run_a)
        sb = _summarize_run(run_b)
        _print_summary(sa)
        _print_summary(sb)
        ok = _compare_runs(sa, sb, tol=float(args.tol))
        print(f"\nCOMPARE_OK: {ok}")
        raise SystemExit(0 if ok else 2)

    results_dir = Path(args.results_dir).resolve()
    run_dirs = _find_run_dirs(results_dir)
    if not run_dirs:
        print(f"No run directories found under: {results_dir}")
        raise SystemExit(1)

    all_ok = True
    for rd in run_dirs:
        s = _summarize_run(rd)
        if (not s.get("ok")) and bool(args.repair):
            _repair_run(rd)
            s = _summarize_run(rd)
        _print_summary(s)
        if not s.get("ok"):
            all_ok = False

    print(f"\nALL_OK: {all_ok}")
    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
