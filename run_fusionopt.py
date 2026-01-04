import argparse
import copy
import os
import random
import shutil
from datetime import datetime

import numpy as np
import yaml

from fusionopt.config import Config
from fusionopt.engine import FusionOptEngine
from fusionopt.adapters_vmec import VmecAdapter
from fusionopt.adapters_gsco import GscoAdapter


def _sanitize_config_for_logging(cfg: dict) -> dict:
    cfg2 = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
    model = cfg2.get("model")
    if isinstance(model, dict):
        model["api_key"] = ""
        if "api_key_file" in model:
            model["api_key_file"] = ""
    return cfg2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config_path
    if not (config_path.startswith("problem/") or os.path.isabs(config_path)):
        config_path = os.path.join("problem", config_path)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    cfg = Config(config_data)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    if "seed" not in config_data or not isinstance(config_data.get("seed"), dict):
        config_data["seed"] = {}
    config_data["seed"]["master"] = seed

    problem_id = cfg.get("protocol.problem_id", None)
    if problem_id is None:
        raise RuntimeError("Missing protocol.problem_id in config")

    algo_id = cfg.get("protocol.algo_id", None) or "fusionopt_v1"

    results_base_dir = cfg.get("logging.results_base_dir", None)
    if results_base_dir is None:
        results_base_dir = os.path.join("./results", str(problem_id))
        if "logging" not in config_data or not isinstance(config_data.get("logging"), dict):
            config_data["logging"] = {}
        config_data["logging"]["results_base_dir"] = results_base_dir

    run_name = cfg.get("exper_name", "run")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_id = args.run_id or f"{run_name}_seed{seed}_{ts}"

    run_dir = os.path.join(results_base_dir, str(algo_id), str(run_id))

    os.makedirs(run_dir, exist_ok=True)

    if problem_id == "stellarator_vmec":
        template_project_path = cfg.get("vmec.template_project_path", None)
        project_path = cfg.get("vmec.project_path", None)
        input_file = cfg.get("vmec.input_file", None)
        if (template_project_path or project_path) and input_file:
            work_dir = os.path.join(run_dir, "vmec_work")
            os.makedirs(work_dir, exist_ok=True)

            src_root = str(template_project_path or project_path)
            try:
                for name in os.listdir(src_root):
                    if name == "backups":
                        continue
                    src_path = os.path.join(src_root, name)
                    dst_path = os.path.join(work_dir, name)
                    if os.path.isdir(src_path):
                        if not os.path.exists(dst_path):
                            shutil.copytree(src_path, dst_path)
                    else:
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
            except FileNotFoundError:
                pass

            if "vmec" not in config_data or not isinstance(config_data.get("vmec"), dict):
                config_data["vmec"] = {}
            config_data["vmec"]["project_path"] = work_dir
            cfg = Config(config_data)

    module_path = cfg.get("evalutor_path")
    if not module_path:
        raise RuntimeError("Missing evalutor_path in config")

    module = __import__(module_path, fromlist=["RewardingSystem", "generate_initial_population"])
    RewardingSystem = getattr(module, "RewardingSystem")
    generate_initial_population = getattr(module, "generate_initial_population")

    reward_system = RewardingSystem(cfg)

    rng = random.Random(seed)
    if problem_id == "stellarator_vmec":
        adapter = VmecAdapter(reward_system, cfg, rng)
    elif problem_id == "stellarator_coil_gsco_lite":
        adapter = GscoAdapter(reward_system, cfg, rng)
    else:
        raise RuntimeError(f"Unsupported problem_id: {problem_id}")

    # Ensure algo_id exists in logged config
    if "protocol" not in config_data or not isinstance(config_data.get("protocol"), dict):
        config_data["protocol"] = {}
    config_data["protocol"]["algo_id"] = algo_id

    sanitized_config_data = _sanitize_config_for_logging(config_data)

    print(f"problem_id: {problem_id}")
    print(f"algo_id: {algo_id}")
    print(f"run_dir: {run_dir}")

    initial_decisions = generate_initial_population(cfg, seed)

    engine = FusionOptEngine(
        config_data=sanitized_config_data,
        config=cfg,
        problem_id=problem_id,
        algo_id=algo_id,
        seed=seed,
        run_id=run_id,
        reward_system=reward_system,
        adapter=adapter,
        goals=cfg.get("goals") or [],
        run_dir=run_dir,
    )

    engine.run(initial_decisions)


if __name__ == "__main__":
    main()
