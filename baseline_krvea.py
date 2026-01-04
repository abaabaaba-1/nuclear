import argparse
import csv
import json
import os
import random
import re
import time
import warnings
from datetime import datetime
from typing import List, Tuple

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from reference_agent import ReferenceAgent

from algorithm.base import ItemFactory
from model.util import nsga2_selection
from eval_logger import EvalLogger

try:
    from problem.stellarator_vmec.vmec_reset_helper import maybe_reset_vmec_inputs
except ImportError:
    def maybe_reset_vmec_inputs(*_args, **_kwargs):
        return


warnings.filterwarnings("ignore", category=ConvergenceWarning)


COEFF_KEY_PATTERN = re.compile(r"([RZ]B[CS])\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.IGNORECASE)


def _parse_mode_numbers(key: str) -> tuple[int | None, int | None]:
     norm_key = key.strip().replace(" ", "")
     match = COEFF_KEY_PATTERN.match(norm_key)
     if not match:
         return None, None
     try:
         return abs(int(match.group(2))), abs(int(match.group(3)))
     except Exception:
         return None, None


class Config:
    """Lightweight config wrapper compatible with existing baseline scripts."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._data
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def to_string(self):
        return yaml.dump(self._data)


class DesignEncoder:
    """Encode VMEC designs as deltas w.r.t. baseline coefficients.

    For each coefficient key k, we encode a scalar feature z_k such that
    z_k = 0  <=>  design uses the baseline value.

    If base_k != 0: z_k = (v_k - base_k) / |base_k|
    If base_k == 0: z_k = v_k

    This matches the semantics in problem/stellarator_vmec/evaluator.py,
    where missing keys mean "use baseline value" rather than 0.
    """

    def __init__(
        self,
        base_coeffs: dict,
        coeff_keys: List[str],
        m_max: int | None = None,
        n_max: int | None = None,
    ):
        self.base = base_coeffs
        self.keys: List[str] = []

        for k in coeff_keys:
            mk, nk = _parse_mode_numbers(k)
            if m_max is not None and n_max is not None:
                if mk is None or nk is None or mk > m_max or nk > n_max:
                    continue
            self.keys.append(k)

        if not self.keys:
            self.keys = list(coeff_keys)

    def encode_delta(self, json_str: str) -> np.ndarray:
        payload = json.loads(json_str)
        delta = payload.get("new_coefficients", {}) or {}
        z = []
        for key in self.keys:
            base_val = self.base.get(key, 0.0)
            v = delta.get(key, base_val)
            if base_val != 0.0:
                z_k = (v - base_val) / abs(base_val)
            else:
                z_k = v
            z.append(float(z_k))
        return np.array(z, dtype=float)


class SurrogateManager:
    """Simple multi-objective Gaussian Process surrogate manager.

    One independent GPR per objective, trained on encoded design features
    and the transformed objective scores (which are already normalized and
    to be minimized).
    """

    def __init__(self, encoder: DesignEncoder, n_obj: int, alpha: float = 1e-3):
        self.encoder = encoder
        self.n_obj = n_obj
        self.X: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []

        dim = len(self.encoder.keys)
        kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))

        self.models = [
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=2,
            )
            for _ in range(n_obj)
        ]

    def add_data(self, items: List):
        for it in items:
            if it.scores is None:
                continue
            x = self.encoder.encode_delta(it.value)
            y = np.asarray(it.scores, dtype=float)
            self.X.append(x)
            self.Y.append(y)

    def fit(self):
        if len(self.X) < 2:
            return
        X = np.vstack(self.X)
        Y = np.vstack(self.Y)
        for j, model in enumerate(self.models):
            model.fit(X, Y[:, j])

    def predict(self, json_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not json_list:
            return np.zeros((0, self.n_obj)), np.zeros((0, self.n_obj))
        X = np.vstack([self.encoder.encode_delta(s) for s in json_list])
        mus, sigmas = [], []
        for model in self.models:
            mu, std = model.predict(X, return_std=True)
            mus.append(mu)
            sigmas.append(std)
        mu = np.stack(mus, axis=1)
        sigma = np.stack(sigmas, axis=1)
        return mu, sigma


def apd_select(
    F: np.ndarray,
    ref_dirs: np.ndarray,
    n_select: int,
    alpha: float = 2.0,
    beta: float = 1.0,
    t: int = 1,
    T: int = 1,
) -> np.ndarray:
    """Angle Penalized Distance (APD) based selection.

    F: (N, M) matrix of objective values to be minimized (e.g., LCB).
    ref_dirs: (K, M) reference directions.

    Returns indices of the n_select best individuals under APD.
    This is a simplified environment selection step inspired by RVEA.
    """

    if F.shape[0] <= n_select:
        return np.arange(F.shape[0], dtype=int)

    N, M = F.shape
    z_min = F.min(axis=0)
    z_max = F.max(axis=0)
    denom = np.where(z_max - z_min < 1e-12, 1.0, z_max - z_min)
    F_norm = (F - z_min) / denom

    norms = np.linalg.norm(F_norm, axis=1) + 1e-12

    ref_norms = np.linalg.norm(ref_dirs, axis=1) + 1e-12
    cos_theta = (F_norm @ ref_dirs.T) / (norms[:, None] * ref_norms[None, :])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Assign each individual to the closest reference direction
    k_idx = theta.argmin(axis=1)
    theta_min = theta[np.arange(N), k_idx]

    # Compute Gamma_j for each reference direction
    K = ref_dirs.shape[0]
    gamma = np.full(K, 1e-12)
    for j in range(K):
        mask = k_idx == j
        if np.any(mask):
            gamma[j] = theta_min[mask].max()

    t_ratio = (t / max(float(T), 1.0)) ** beta
    apd = norms * (1.0 + alpha * t_ratio * theta_min / gamma[k_idx])

    return np.argsort(apd)[:n_select]


def generate_candidates_vmec(
    base_coeffs: dict,
    config: Config,
    evaluator_module,
    n_candidates: int,
    rng: random.Random,
) -> List[str]:
    """Generate JSON candidates for VMEC using the same mutation logic
    as the VMEC evaluator's initial population generator.
    """

    mutate_seed = getattr(evaluator_module, "_mutate_seed_coefficients", None)
    extract_delta = getattr(evaluator_module, "_extract_delta_coefficients", None)

    if mutate_seed is None or extract_delta is None:
        # Fallback: simple Gaussian perturbations of a small subset of coefficients
        keys = list(base_coeffs.keys())
        max_changes = config.get("llm_constraints.max_coeff_changes", 12)
        low = config.get("llm_constraints.low_order_max_rel_change", 0.02)
        high = config.get("llm_constraints.high_order_max_rel_change", 0.05)

        candidates = []
        for _ in range(n_candidates):
            num_to_mutate = rng.randint(1, max_changes)
            chosen = rng.sample(keys, num_to_mutate)
            new_coeffs = {}
            for k in chosen:
                base_val = base_coeffs.get(k, 0.0)
                if base_val == 0.0:
                    continue
                rel = rng.uniform(-high, high)
                new_coeffs[k] = base_val * (1.0 + rel)
            if not new_coeffs:
                continue
            candidates.append(json.dumps({"new_coefficients": new_coeffs}))
        return candidates

    # Preferred path: reuse evaluator's helper functions
    candidates = []
    max_changes = config.get("llm_constraints.max_coeff_changes", 12)
    for _ in range(n_candidates):
        mutated = mutate_seed(base_coeffs)
        delta = extract_delta(base_coeffs, mutated)
        if not delta:
            continue
        if max_changes and len(delta) > max_changes:
            keys = list(delta.keys())
            rng.shuffle(keys)
            keys = keys[:max_changes]
            delta = {k: delta[k] for k in keys}
        candidates.append(json.dumps({"new_coefficients": delta}))
    return candidates


def generate_candidates_generic(
    base_coeffs: dict,
    config: Config,
    evaluator_module,
    n_candidates: int,
    rng: random.Random,
) -> List[str]:
    mutate_seed = getattr(evaluator_module, "_mutate_seed_coefficients", None)
    extract_delta = getattr(evaluator_module, "_extract_delta_coefficients", None)

    if mutate_seed is None or extract_delta is None:
        return []

    candidates: List[str] = []
    max_changes = config.get("llm_constraints.max_coeff_changes", None)
    for _ in range(int(n_candidates)):
        mutated = mutate_seed(base_coeffs)
        delta = extract_delta(base_coeffs, mutated)
        if not delta:
            continue
        if max_changes:
            try:
                max_changes_int = int(max_changes)
            except Exception:
                max_changes_int = None
            if max_changes_int and len(delta) > max_changes_int:
                keys = list(delta.keys())
                rng.shuffle(keys)
                keys = keys[:max_changes_int]
                delta = {k: delta[k] for k in keys}
        candidates.append(json.dumps({"new_coefficients": delta}))
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Run Kriging-assisted RVEA-style baseline (K-RVEA-like) for VMEC",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="sacs_geo_jk/config.yaml",
        help="Path to config YAML (e.g., problem/stellarator_vmec/config_moead.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Load configuration
    config_path = args.config
    if not (config_path.startswith("problem/") or os.path.isabs(config_path)):
        config_path = os.path.join("problem", config_path)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    config = Config(config_data)
    seed = args.seed

    if 'protocol' not in config_data or not isinstance(config_data.get('protocol'), dict):
        config_data['protocol'] = {}
    config_data['protocol'].setdefault('algo_id', 'k_rvea')

    if 'logging' not in config_data or not isinstance(config_data.get('logging'), dict):
        config_data['logging'] = {}
    problem_id = config.get('protocol.problem_id', None)
    if problem_id:
        config_data['logging']['results_base_dir'] = os.path.join('./results', str(problem_id))

    # 2. Setup environment
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    property_list = config.get("goals")
    n_obj = len(property_list)

    module_path = config.get("evalutor_path")
    module = __import__(module_path, fromlist=["RewardingSystem", "generate_initial_population"])
    RewardingSystem = getattr(module, "RewardingSystem")
    generate_initial_population = getattr(module, "generate_initial_population")

    eval_logger = None
    run_dir = None
    evalutor_path = config.get('evalutor_path')
    project_path = config.get('vmec.project_path')
    maybe_reset_vmec_inputs(evalutor_path, project_path, 'pre')
    try:
        problem_id = config.get('protocol.problem_id', None)
        results_base_dir = os.path.join('./results', str(problem_id)) if problem_id else None
        algo_id = config.get('protocol.algo_id', None)
        if problem_id and results_base_dir and algo_id:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            run_name = config.get('exper_name', 'run')
            run_id = f"{run_name}_seed{seed}_{ts}"
            run_dir = os.path.join(results_base_dir, algo_id, run_id)
            eval_logger = EvalLogger(run_dir, problem_id, algo_id, seed, run_id, config_data, property_list)

        # NOTE: this baseline is currently tailored for the VMEC problem, where
        # RewardingSystem exposes .base_coeffs and the evaluator defines
        # _mutate_seed_coefficients/_extract_delta_coefficients.

        reward_system = RewardingSystem(config)
        base_coeffs = reward_system.base_coeffs

        item_factory = ItemFactory(property_list)

        eval_budget = config.get("optimization.eval_budget")
        log_freq = config.get("optimization.log_freq", 50)

        pop_size = config.get("optimization.pop_size")
        try:
            pop_size = int(pop_size) if pop_size is not None else None
        except Exception:
            pop_size = None

        kr_cfg = config.get("krvea", {}) or {}
        n_init = kr_cfg.get("n_init", None)
        n_infill = kr_cfg.get("n_infill", None)
        n_candidates = kr_cfg.get("n_candidates", None)
        n_partitions = kr_cfg.get("n_partitions", 12)
        kappa = kr_cfg.get("kappa", 2.0)
        alpha = kr_cfg.get("alpha", 2.0)
        beta = kr_cfg.get("beta", 1.0)
        gp_alpha = kr_cfg.get("gp_alpha", 1e-3)
        max_train_size = kr_cfg.get("max_train_size", None)
        feature_m_max = kr_cfg.get("feature_m_max", 2)
        feature_n_max = kr_cfg.get("feature_n_max", 1)

        try:
            eval_budget = int(eval_budget)
        except Exception:
            raise ValueError(f"Invalid eval_budget: {eval_budget}")
        if eval_budget <= 0:
            raise ValueError(f"Invalid eval_budget: {eval_budget}")

        try:
            log_freq = int(log_freq)
        except Exception:
            log_freq = 50
        if log_freq <= 0:
            log_freq = 50

        def _as_pos_int(val, default: int) -> int:
            try:
                v = int(val)
                if v > 0:
                    return v
            except Exception:
                pass
            return int(default)

        # 3. Initial design (true evaluations)
        init_jsons_all = generate_initial_population(config, seed)
        if not init_jsons_all:
            raise RuntimeError("Initial population generator returned no candidates.")

        if pop_size is None or pop_size <= 0:
            pop_size = len(init_jsons_all)
        pop_size = int(pop_size)

        n_init = _as_pos_int(n_init, pop_size)
        n_infill = _as_pos_int(n_infill, pop_size)
        n_candidates = _as_pos_int(n_candidates, 5 * pop_size)
        n_partitions = _as_pos_int(n_partitions, 12)

        print(f"Generating initial population (n_init={n_init}) for K-RVEA baseline...")
        init_jsons = init_jsons_all
        if len(init_jsons) < n_init:
            extra_needed = n_init - len(init_jsons)
            print(f"Warning: only {len(init_jsons)} initial candidates, need {n_init}. Reusing with jitter.")
            init_jsons = (init_jsons * ((extra_needed // len(init_jsons)) + 1))[:n_init]
        elif len(init_jsons) > n_init:
            init_jsons = init_jsons[:n_init]

        archive: List[Tuple[object, int]] = []  # (Item, eval_idx)
        eval_idx = 0
        gen_counter = 0

        def eval_batch(json_list: List[str], generation: int) -> List[object]:
            nonlocal eval_idx
            if not json_list:
                return []
            items = [item_factory.create(s) for s in json_list]
            t0 = time.time()
            evaluated, _ = reward_system.evaluate(items)
            dt = time.time() - t0
            for it in evaluated:
                eval_idx += 1
                archive.append((it, eval_idx))

            if eval_logger is not None:
                eval_logger.log_batch(evaluated, generation=int(generation), total_time_sec=dt, tier="true")
            print(f"\rProgress: {eval_idx}/{eval_budget}", end="", flush=True)
            return evaluated

        init_items = eval_batch(init_jsons, generation=gen_counter)

        # 4. Surrogate setup
        feature_keys = sorted(base_coeffs.keys())
        encoder = DesignEncoder(
            base_coeffs,
            feature_keys,
            m_max=feature_m_max,
            n_max=feature_n_max,
        )
        surr = SurrogateManager(encoder, n_obj, alpha=gp_alpha)
        surr.add_data(init_items)

        if max_train_size is not None:
            try:
                max_train_size = int(max_train_size)
            except Exception:
                max_train_size = None
            if max_train_size is not None and max_train_size <= 0:
                max_train_size = None

        # 5. Reference directions
        ref_dirs = None
        try:
            ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=int(n_partitions))
        except Exception:
            ref_dirs = None

        if ref_dirs is None or len(ref_dirs) != pop_size:
            try:
                ref_dirs = get_reference_directions("energy", n_obj, n_points=int(pop_size))
            except Exception:
                ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=int(n_partitions))

        ref_agent = ReferenceAgent(config, n_obj=n_obj, pop_size=pop_size, seed=seed, run_dir=run_dir)
        ref_agent.log_initial(ref_dirs, generation=gen_counter, eval_idx=eval_idx, eval_budget=eval_budget)

        ref_occ_path = os.path.join(run_dir, "ref_occupancy.csv") if run_dir else None
        ref_occ_fieldnames = [
            "generation",
            "eval_idx",
            "eval_budget",
            "n_obj",
            "pop_size",
            "n_feasible",
            "n_nd",
            "occupancy_nonzero_frac",
            "occupancy_zero_frac",
            "occupancy_min",
            "occupancy_max",
            "occupancy_mean",
            "occupancy_std",
        ]

        def _log_ref_occupancy(generation: int):
            if not ref_occ_path:
                return
            try:
                items_only = [it for it, _ in archive]
                F = ref_agent._collect_feasible_F(items_only)
                if F.shape[0] > 0:
                    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    F_nd = F[nd_idx]
                else:
                    F_nd = np.zeros((0, n_obj), dtype=float)

                occ = ref_agent._compute_occupancy(ref_dirs, F_nd)

                row = {
                    "generation": int(generation),
                    "eval_idx": int(eval_idx),
                    "eval_budget": int(eval_budget),
                    "n_obj": int(n_obj),
                    "pop_size": int(pop_size),
                    "n_feasible": int(F.shape[0]),
                    "n_nd": int(F_nd.shape[0]),
                    "occupancy_nonzero_frac": "",
                    "occupancy_zero_frac": "",
                    "occupancy_min": "",
                    "occupancy_max": "",
                    "occupancy_mean": "",
                    "occupancy_std": "",
                }
                if occ is not None and occ.size:
                    row["occupancy_nonzero_frac"] = float(np.mean(occ > 0))
                    row["occupancy_zero_frac"] = float(np.mean(occ == 0))
                    row["occupancy_min"] = int(np.min(occ))
                    row["occupancy_max"] = int(np.max(occ))
                    row["occupancy_mean"] = float(np.mean(occ))
                    row["occupancy_std"] = float(np.std(occ))

                write_header = not os.path.exists(ref_occ_path)
                with open(ref_occ_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=ref_occ_fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception:
                return

        _log_ref_occupancy(generation=int(gen_counter))

        start_time = time.time()

        # 6. Main K-RVEA-style outer loop
        while eval_idx < eval_budget:
            gen_counter += 1

            ref_dirs = ref_agent.maybe_update(
                ref_dirs,
                [it for it, _ in archive],
                generation=gen_counter,
                eval_idx=eval_idx,
                eval_budget=eval_budget,
            )
            # Fit surrogate if enough data
            if max_train_size is not None and len(surr.X) > max_train_size:
                surr.X = surr.X[-max_train_size:]
                surr.Y = surr.Y[-max_train_size:]

            if len(surr.X) >= 2:
                surr.fit()

            # Generate candidate JSONs
            seen_values = {it.value for it, _ in archive}
            cand_jsons: List[str] = []
            max_attempts = 6
            for attempt in range(max_attempts):
                if evalutor_path and str(evalutor_path).startswith("problem.stellarator_vmec"):
                    cand_jsons = generate_candidates_vmec(
                        base_coeffs,
                        config,
                        module,
                        int(n_candidates) * int(attempt + 1),
                        rng,
                    )
                else:
                    cand_jsons = generate_candidates_generic(
                        base_coeffs,
                        config,
                        module,
                        int(n_candidates) * int(attempt + 1),
                        rng,
                    )
                if cand_jsons:
                    cand_jsons = [s for s in cand_jsons if s not in seen_values]
                if cand_jsons:
                    break

            if not cand_jsons:
                print("No new surrogate candidates could be generated after retries; stopping.")
                break

            mu, sigma = surr.predict(cand_jsons)
            if mu.shape[0] == 0:
                print("Surrogate prediction returned empty; stopping.")
                break

            lcb = mu - kappa * sigma

            remaining = eval_budget - eval_idx
            n_select = min(n_infill, remaining, lcb.shape[0])
            selected_idx = apd_select(
                lcb,
                ref_dirs,
                n_select=n_select,
                alpha=alpha,
                beta=beta,
                t=eval_idx,
                T=eval_budget,
            )
            selected_jsons = [cand_jsons[i] for i in selected_idx]

            evaluated_items = eval_batch(selected_jsons, generation=gen_counter)
            surr.add_data(evaluated_items)
            _log_ref_occupancy(generation=int(gen_counter))
        print()
        running_time = time.time() - start_time
        print(f"K-RVEA-like baseline finished in {running_time / 3600:.2f} hours.")
    finally:
        try:
            if eval_logger is not None:
                eval_logger.close()
        except Exception:
            pass
        maybe_reset_vmec_inputs(evalutor_path, project_path, 'post')


if __name__ == "__main__":
    main()
