from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from algorithm.base import ItemFactory
from eval_logger import EvalLogger
from model.util import nsga2_selection

from .json_utils import canonicalize_json_str, md5_hex
from .llm_semop import LlmSemOpManager
from .protocol import ensure_protocol_fields, make_penalty_item


@dataclass
class EngineResult:
    run_dir: str
    n_true_evals: int


class FusionOptEngine:
    def __init__(
        self,
        config_data: dict,
        config,
        problem_id: str,
        algo_id: str,
        seed: int,
        run_id: str,
        reward_system,
        adapter,
        goals: List[str],
        run_dir: str,
    ):
        self.config_data = config_data
        self.config = config
        self.problem_id = problem_id
        self.algo_id = algo_id
        self.seed = int(seed)
        self.run_id = str(run_id)
        self.reward_system = reward_system
        self.adapter = adapter
        self.goals = list(goals or [])
        self.run_dir = run_dir

        self.item_factory = ItemFactory(self.goals)

        self.pop_size = int(self.config.get("optimization.pop_size"))
        self.eval_budget = int(self.config.get("optimization.eval_budget"))
        self.log_freq = int(self.config.get("optimization.log_freq", 50) or 50)

        self.offspring_per_gen = int(self.config.get("fusionopt.offspring_per_gen", self.pop_size) or self.pop_size)
        self.batch_eval_size = int(self.config.get("fusionopt.batch_eval_size", self.pop_size) or self.pop_size)
        self.max_attempts_per_child = int(self.config.get("fusionopt.max_attempts_per_child", 50) or 50)

        self.dedup_scope = str(self.config.get("fusionopt.gate.dedup_scope", "run") or "run")

        ow = self.config.get("fusionopt.operator_weights", None) or {}
        self.w_cx = float(ow.get("std_crossover", 0.7) or 0.7)
        self.w_mut = float(ow.get("std_mutation", 0.3) or 0.3)
        self.w_rsp = float(ow.get("std_resample", 0.0) or 0.0)
        self.w_llm = float(ow.get("llm_semop", 0.0) or 0.0)

        self.use_heu_repair = bool(self.config.get("fusionopt.use_heu_repair", True))

        # Dedicated RNG to avoid perturbing operator sampling / adapter RNG when using probabilistic repair.
        self.heu_repair_rng = random.Random(self.seed + 99173)

        self.seen_ids = set()

        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)
        # Phase 4 placeholder
        llm_calls_path = os.path.join(self.run_dir, "logs", "llm_calls.jsonl")
        if not os.path.exists(llm_calls_path):
            try:
                with open(llm_calls_path, "w"):
                    pass
            except Exception:
                pass

        self.llm_semop = LlmSemOpManager(
            config=self.config,
            run_dir=self.run_dir,
            problem_id=self.problem_id,
            adapter=self.adapter,
            seed=self.seed,
        )
        if not getattr(self.llm_semop, "enabled", False):
            self.w_llm = 0.0

        self.eval_logger = EvalLogger(
            self.run_dir,
            self.problem_id,
            self.algo_id,
            self.seed,
            self.run_id,
            self.config_data,
            self.goals,
        )

    def _sample_operator(self) -> str:
        total = max(0.0, self.w_cx) + max(0.0, self.w_mut) + max(0.0, self.w_rsp) + max(0.0, self.w_llm)
        if total <= 0.0:
            return "std_mutation"
        r = random.random() * total
        if r < self.w_cx:
            return "std_crossover"
        r -= self.w_cx
        if r < self.w_mut:
            return "std_mutation"
        r -= self.w_mut
        if r < self.w_rsp:
            return "std_resample"
        return "llm_semop"

    def _sample_std_operator(self) -> str:
        total = max(0.0, self.w_cx) + max(0.0, self.w_mut) + max(0.0, self.w_rsp)
        if total <= 0.0:
            return "std_mutation"
        r = random.random() * total
        if r < self.w_cx:
            return "std_crossover"
        r -= self.w_cx
        if r < self.w_mut:
            return "std_mutation"
        return "std_resample"

    def _gate_and_repair(self, decision_json: str) -> Optional[str]:
        gated = self.adapter.gate(decision_json)
        if gated is None:
            return None
        if self.use_heu_repair:
            p = float(self.config.get("fusionopt.heu_repair.prob", 1.0) or 1.0)
            if p >= 1.0:
                do_repair = True
            elif p <= 0.0:
                do_repair = False
            else:
                do_repair = self.heu_repair_rng.random() < p

            if do_repair:
                repaired = self.adapter.heu_repair(gated)
                if repaired is None:
                    return None
                return repaired
        # canonicalize anyway for stable hashing
        return canonicalize_json_str(gated)

    def _dedup_ok(self, decision_json: str, generation: int) -> bool:
        if self.dedup_scope == "generation":
            # generation-scope handled by caller
            return True
        did = md5_hex(decision_json)
        if did in self.seen_ids:
            return False
        self.seen_ids.add(did)
        return True

    def _evaluate_decisions(self, decisions: List[str], generation: int) -> List:
        if not decisions:
            return []
        items_in = [self.item_factory.create(s) for s in decisions]

        t0 = time.time()
        evaluated_items, _log = self.reward_system.evaluate(items_in)
        dt = time.time() - t0

        # Some evaluators may drop invalid/repeated items, and may also mutate item.value.
        # Use object identity rather than decision_json string matching.
        evaluated_set = set(evaluated_items or [])
        out_items = []
        for idx, s in enumerate(decisions):
            it_in = items_in[idx]
            if it_in in evaluated_set:
                out_items.append(it_in)
            else:
                out_items.append(make_penalty_item(self.item_factory, s, self.goals))

        ensure_protocol_fields(self.problem_id, out_items)

        self.eval_logger.log_batch(out_items, generation=generation, total_time_sec=dt, tier="true")
        return out_items

    def run(self, initial_decisions: List[str]) -> EngineResult:
        try:
            # 1) init population
            population_jsons = []
            gen_seen = set()
            attempts = 0
            while len(population_jsons) < self.pop_size and attempts < self.pop_size * self.max_attempts_per_child:
                if attempts < len(initial_decisions):
                    cand = initial_decisions[attempts]
                else:
                    cand = self.adapter.std_resample()

                cand2 = self._gate_and_repair(cand)
                attempts += 1
                if cand2 is None:
                    continue

                if self.dedup_scope == "generation":
                    if cand2 in gen_seen:
                        continue
                    gen_seen.add(cand2)
                else:
                    if not self._dedup_ok(cand2, generation=0):
                        continue

                population_jsons.append(cand2)

            if len(population_jsons) < self.pop_size:
                raise RuntimeError(f"Failed to build initial population: {len(population_jsons)}/{self.pop_size}")

            population = self._evaluate_decisions(population_jsons, generation=0)
            n_true_evals = len(population)

            generation = 0
            while n_true_evals < self.eval_budget:
                generation += 1
                remaining = self.eval_budget - n_true_evals
                if remaining <= 0:
                    break

                if self.w_llm > 0.0:
                    self.llm_semop.maybe_schedule_batch(generation=generation, population=population)

                target = min(self.offspring_per_gen, remaining)

                offspring = []
                gen_seen = set()
                max_attempts = self.max_attempts_per_child * max(1, target)
                attempts = 0

                while len(offspring) < target and attempts < max_attempts:
                    op = self._sample_operator()

                    child = None
                    llm_meta = None
                    if op == "llm_semop":
                        cand = self.llm_semop.try_get_candidate()
                        if cand is not None:
                            child = cand.decision_json
                            llm_meta = cand.meta
                        else:
                            op = self._sample_std_operator()

                    if child is None:
                        if op == "std_resample":
                            child = self.adapter.std_resample()
                        elif op == "std_mutation":
                            p = random.choice(population)
                            child = self.adapter.std_mutation(getattr(p, "value", ""))
                        else:
                            pa = random.choice(population)
                            pb = random.choice(population)
                            child = self.adapter.std_crossover(getattr(pa, "value", ""), getattr(pb, "value", ""))

                    child2 = self._gate_and_repair(child)
                    attempts += 1
                    if child2 is None:
                        if llm_meta is not None:
                            try:
                                self.llm_semop.log_validation_failed(
                                    generation=generation,
                                    meta=llm_meta,
                                    error="engine_gate_or_repair_failed",
                                )
                            except Exception:
                                pass
                        continue

                    if self.dedup_scope == "generation":
                        if child2 in gen_seen:
                            continue
                        gen_seen.add(child2)
                    else:
                        if not self._dedup_ok(child2, generation=generation):
                            continue

                    offspring.append(child2)

                if not offspring:
                    raise RuntimeError(
                        "Gate/repair too strict: no offspring generated. "
                        "Consider relaxing gate or increasing max_attempts_per_child."
                    )

                # evaluate in batches (optional)
                offspring_items = []
                start = 0
                while start < len(offspring):
                    batch = offspring[start : start + self.batch_eval_size]
                    evaluated = self._evaluate_decisions(batch, generation=generation)
                    offspring_items.extend(evaluated)
                    n_true_evals += len(evaluated)
                    start += self.batch_eval_size

                # environmental selection
                pool = population + offspring_items
                population = nsga2_selection(pool, self.pop_size)

            return EngineResult(run_dir=self.run_dir, n_true_evals=n_true_evals)
        finally:
            try:
                self.llm_semop.close()
            except Exception:
                pass
            self.eval_logger.close()
