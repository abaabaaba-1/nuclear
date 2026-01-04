import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.util import cal_hv


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _resolve_results_base_dir(config_data: dict) -> tuple[str, str]:
    protocol = config_data.get("protocol") or {}
    problem_id = protocol.get("problem_id") or "unknown_problem"

    logging_cfg = config_data.get("logging") or {}
    results_base_dir = logging_cfg.get("results_base_dir")

    if not results_base_dir:
        results_base_dir = os.path.join("./results", str(problem_id))

    return str(problem_id), str(results_base_dir)


def _write_run_meta(run_dir: str, problem_id: str, algo_id: str, run_id: str, seed: int, config_data: dict) -> None:
    meta = {
        "problem_id": problem_id,
        "algo_id": algo_id,
        "run_id": run_id,
        "seed": int(seed),
        "seed.master": (config_data.get("seed", {}) or {}).get("master", int(seed)),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _safe_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(v, default=0):
    f = _safe_float(v)
    if f is None:
        return int(default)
    try:
        return int(f)
    except Exception:
        return int(default)


def _postprocess_metrics(run_dir: str, config_data: dict, seed: int) -> None:
    csv_path = os.path.join(run_dir, "evaluations.csv")
    if not os.path.exists(csv_path):
        return

    goals = config_data.get("goals") or []
    num_obj = len(goals)
    if num_obj <= 0:
        return

    records = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feasible = _safe_int(row.get("feasible", "0"), default=0)
                status = str(row.get("status", ""))
                if feasible != 1 or status != "ok":
                    continue

                total = _safe_float(row.get("total"))

                scores = []
                ok = True
                for i in range(num_obj):
                    v = _safe_float(row.get(f"f{i+1}_min"))
                    if v is None:
                        ok = False
                        break
                    scores.append(v)
                if not ok:
                    continue

                if total is None:
                    try:
                        total = 1.0 - float(np.mean(np.asarray(scores, dtype=float)))
                    except Exception:
                        total = 0.0

                records.append(
                    {
                        "generation": _safe_int(row.get("generation", "0"), default=0),
                        "eval_id": _safe_int(row.get("eval_id", "0"), default=0),
                        "decision_id": str(row.get("decision_id", "")),
                        "total": float(total),
                        "scores": scores,
                    }
                )
    except Exception:
        return

    if not records:
        return

    generations = sorted(set(int(r["generation"]) for r in records))
    results = []

    for g in generations:
        subset = [r for r in records if int(r["generation"]) <= int(g)]
        if not subset:
            continue

        subset_sorted = sorted(subset, key=lambda r: float(r.get("total", 0.0)), reverse=True)
        generated_num = len(subset_sorted)
        unique_decisions = len(set(r.get("decision_id", "") for r in subset_sorted if r.get("decision_id")))

        top1 = float(subset_sorted[0]["total"]) if subset_sorted else 0.0
        top100 = subset_sorted[: min(100, len(subset_sorted))]

        hv = 0.0
        if len(top100) >= 2:
            try:
                hv = float(cal_hv(np.asarray([r["scores"] for r in top100], dtype=float)))
            except Exception:
                hv = 0.0

        results.append(
            {
                "Training_step": int(g),
                "generated_num": int(generated_num),
                "all_unique_moles": int(unique_decisions),
                "avg_top1": float(top1),
                "hypervolume": float(hv),
            }
        )

    out_obj = {
        "results": results,
        "params": "",
    }

    out_path = os.path.join(run_dir, "metrics.json")
    try:
        with open(out_path, "w") as f:
            json.dump(out_obj, f, indent=2)
    except Exception:
        return

    try:
        problem_id = (config_data.get("protocol") or {}).get("problem_id") or "unknown_problem"
        results_base_dir = (config_data.get("logging") or {}).get("results_base_dir") or os.path.join("./results", str(problem_id))
        if not os.path.isabs(results_base_dir):
            results_base_dir = str((REPO_ROOT / results_base_dir).resolve())
        seed = int(seed)
        baselines_dir = os.path.join(results_base_dir, "baselines")
        os.makedirs(baselines_dir, exist_ok=True)
        legacy_out = os.path.join(baselines_dir, f"OpenEvolve_{seed}_metrics.json")
        with open(legacy_out, "w") as f:
            json.dump(out_obj, f, indent=2)
    except Exception:
        return


def _parse_cmd(args: argparse.Namespace) -> list[str]:
    if args.cmd_from_env:
        raw = os.environ.get(args.cmd_from_env, "").strip()
        if not raw:
            raise ValueError(f"Environment variable {args.cmd_from_env} is empty")
        return shlex.split(raw)

    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("No Open Evolve command provided. Use `--cmd-from-env OPEN_EVOLVE_CMD` or append `-- <command ...>`")
    return cmd


_ENV_REF_RE = re.compile(r"\$(\{)?([A-Za-z_][A-Za-z0-9_]*)\}?\b")


def _expand_env_tokens(cmd: list[str], env: dict[str, str]) -> list[str]:
    expanded: list[str] = []
    for tok in cmd:
        if not isinstance(tok, str):
            expanded.append(tok)
            continue

        def repl(match: re.Match) -> str:
            var = match.group(2)
            return str(env.get(var, match.group(0)))

        expanded.append(_ENV_REF_RE.sub(repl, tok))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to problem config yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo-id", default="open_evolve")
    parser.add_argument("--cmd-from-env", default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str((REPO_ROOT / config_path).resolve())

    config_data = _load_yaml(config_path)
    problem_id, results_base_dir = _resolve_results_base_dir(config_data)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_id = f"{args.algo_id}_seed{args.seed}_{ts}"
    run_dir = os.path.join(results_base_dir, args.algo_id, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)
    except Exception:
        pass

    _write_run_meta(run_dir, problem_id, args.algo_id, run_id, args.seed, config_data)

    cmd = _parse_cmd(args)

    env = os.environ.copy()
    env["MOLLM_RUN_DIR"] = os.path.abspath(run_dir)
    env["MOLLM_REPO_ROOT"] = str(REPO_ROOT)
    env["MOLLM_PROBLEM_ID"] = str(problem_id)
    env["MOLLM_ALGO_ID"] = str(args.algo_id)
    env["MOLLM_RUN_ID"] = str(run_id)
    env["MOLLM_SEED"] = str(args.seed)
    env["MOLLM_CONFIG_PATH"] = os.path.abspath(config_path)

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(REPO_ROOT)

    cmd = _expand_env_tokens(cmd, env)

    with open(os.path.join(run_dir, "command.json"), "w") as f:
        json.dump({"cmd": cmd, "cwd": str(REPO_ROOT)}, f, indent=2)

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=out_f, stderr=err_f)

    if proc.returncode != 0:
        raise RuntimeError(f"Open Evolve failed with exit code {proc.returncode}. See logs in {run_dir}")

    _postprocess_metrics(run_dir, config_data, seed=args.seed)


if __name__ == "__main__":
    main()
