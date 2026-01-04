import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

repo_root = os.environ.get("MOLLM_REPO_ROOT")
if not repo_root:
    try:
        repo_root = str(Path(__file__).resolve().parents[3])
    except Exception:
        repo_root = None
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from openevolve.evaluation_result import EvaluationResult

from algorithm.base import ItemFactory
from eval_logger import EvalLogger
from problem.stellarator_vmec.evaluator import RewardingSystem


_CALL_INDEX = 0
_EVAL_LOGGER = None
_EVAL_LOGGER_RUN_DIR = None
_REWARD_SYSTEM = None
_REWARD_SYSTEM_SEED = None


class _ConfigWrapper:
    def __init__(self, data: dict):
        self.data = data or {}

    def get(self, key: str, default=None):
        if not isinstance(key, str):
            return default
        cur: Any = self.data
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _infer_run_dir(program_path: str) -> Optional[str]:
    try:
        p = Path(program_path).resolve()
    except Exception:
        return None
    for parent in [p.parent] + list(p.parents):
        if (parent / "run_meta.json").exists():
            return str(parent)
        if (parent / "command.json").exists():
            return str(parent)
    return None


def _infer_run_dir_from_config(config_data: dict, algo_id: str, seed: int) -> Optional[str]:
    if not isinstance(config_data, dict):
        return None
    protocol = config_data.get("protocol") or {}
    problem_id = protocol.get("problem_id") or "unknown_problem"
    logging_cfg = config_data.get("logging") or {}
    results_base_dir = logging_cfg.get("results_base_dir")
    if not results_base_dir:
        results_base_dir = os.path.join("./results", str(problem_id))

    base_dir = Path(results_base_dir)
    if not base_dir.is_absolute():
        rr = os.environ.get("MOLLM_REPO_ROOT")
        if rr:
            base_dir = (Path(rr) / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()

    algo_dir = base_dir / str(algo_id)
    if not algo_dir.exists() or not algo_dir.is_dir():
        return None

    candidates = []
    try:
        for p in algo_dir.iterdir():
            if not p.is_dir():
                continue
            if f"seed{int(seed)}" not in p.name:
                continue
            candidates.append(p)
    except Exception:
        candidates = []

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def evaluate(program_path: str):
    config_path = os.environ.get("MOLLM_CONFIG_PATH")
    if not config_path or not os.path.exists(config_path):
        return EvaluationResult(
            metrics={"score": 0.0, "error": 1.0},
            artifacts={
                "error_type": "MissingConfig",
                "error_message": "MOLLM_CONFIG_PATH not set or file does not exist",
            },
        )

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}

    cfg = _ConfigWrapper(config_data)

    seed_env = int(os.environ.get("MOLLM_SEED", "42"))
    rng = random.Random(seed_env)

    goals = cfg.get("goals", ["volume", "aspect_ratio", "magnetic_shear"]) or [
        "volume",
        "aspect_ratio",
        "magnetic_shear",
    ]

    batch_n = int(os.environ.get("OPEN_EVOLVE_BATCH_N", "4"))

    try:
        program = _load_program(program_path)
        if not hasattr(program, "propose"):
            return EvaluationResult(
                metrics={"score": 0.0, "error": 1.0},
                artifacts={
                    "error_type": "MissingFunction",
                    "error_message": "Program must define propose(rng, n, base_coeffs, incumbent_json='', ...)",
                },
            )

        algo_id = os.environ.get("MOLLM_ALGO_ID", "open_evolve")
        run_id = os.environ.get("MOLLM_RUN_ID", "")
        problem_id = os.environ.get("MOLLM_PROBLEM_ID", "stellarator_vmec")

        run_dir = os.environ.get("MOLLM_RUN_DIR")
        if not run_dir:
            run_dir = _infer_run_dir(program_path)
        if not run_dir:
            run_dir = _infer_run_dir_from_config(config_data, algo_id=algo_id, seed=seed_env)

        global _EVAL_LOGGER
        global _EVAL_LOGGER_RUN_DIR
        if run_dir and (_EVAL_LOGGER is None or _EVAL_LOGGER_RUN_DIR != run_dir):
            try:
                _EVAL_LOGGER = EvalLogger(
                    run_dir=run_dir,
                    problem_id=problem_id,
                    algo_id=algo_id,
                    seed=seed_env,
                    run_id=run_id,
                    config_data=config_data,
                    goals=list(goals),
                )
                _EVAL_LOGGER_RUN_DIR = run_dir
            except Exception:
                try:
                    if run_dir:
                        with open(os.path.join(run_dir, "eval_logger_error.txt"), "a") as f:
                            f.write("EvalLogger init failed\n")
                except Exception:
                    pass
                _EVAL_LOGGER = None
                _EVAL_LOGGER_RUN_DIR = None

        global _REWARD_SYSTEM
        global _REWARD_SYSTEM_SEED
        if _REWARD_SYSTEM is None or _REWARD_SYSTEM_SEED != seed_env:
            _REWARD_SYSTEM = RewardingSystem(cfg)
            _REWARD_SYSTEM_SEED = seed_env

        base_coeffs = getattr(_REWARD_SYSTEM, "base_coeffs", {}) or {}

        global _CALL_INDEX
        call_idx = int(_CALL_INDEX)
        _CALL_INDEX += 1

        proposals = program.propose(
            rng=rng,
            n=batch_n,
            base_coeffs=base_coeffs,
            incumbent_json="",
            max_coeff_changes=int(cfg.get("llm_constraints.max_coeff_changes", 12)),
            low_order_max_rel_change=float(cfg.get("llm_constraints.low_order_max_rel_change", 0.02)),
            high_order_max_rel_change=float(cfg.get("llm_constraints.high_order_max_rel_change", 0.05)),
        )

        item_factory = ItemFactory(list(goals))
        items = []
        for cand in proposals:
            if isinstance(cand, dict):
                value = json.dumps(cand)
            else:
                value = str(cand)
            items.append(item_factory.create(value))

        t0 = time.time()
        items, _ = _REWARD_SYSTEM.evaluate(items)
        dt = time.time() - t0

        if _EVAL_LOGGER is not None:
            try:
                _EVAL_LOGGER.log_batch(items, generation=call_idx, total_time_sec=dt, tier="true")
            except Exception:
                pass

        best_total = None
        best_value = ""
        totals = []
        for it in items:
            if it.total is None:
                continue
            totals.append(float(it.total))
            if best_total is None or float(it.total) > best_total:
                best_total = float(it.total)
                best_value = it.value

        if best_total is None:
            best_total = 0.0

        metrics = {
            "score": float(best_total),
            "combined_score": float(best_total),
            "best_total": float(best_total),
            "avg_total": float(sum(totals) / max(len(totals), 1)),
            "n_scored": float(len(totals)),
            "eval_time_sec": float(dt),
        }

        return EvaluationResult(metrics=metrics, artifacts={"best_candidate_json": best_value or ""})

    except Exception as e:
        return EvaluationResult(
            metrics={"score": 0.0, "error": 1.0},
            artifacts={"error_type": "Exception", "error_message": str(e)[:500]},
        )
