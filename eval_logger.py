import csv
import hashlib
import json
import os
from datetime import datetime

import atexit

import numpy as np
import yaml


class EvalLogger:
    """Minimal evaluation logger for Phase 0 protocol.

    Writes a CSV file evaluations.csv under a protocol-aligned run directory.
    This logger is intentionally lightweight and only used by GSCO-Lite
    baselines in Phase 0, without changing their selection logic.
    """

    def __init__(self, run_dir, problem_id, algo_id, seed, run_id, config_data, goals):
        self.run_dir = run_dir
        self.problem_id = problem_id
        self.algo_id = algo_id
        self.seed = seed
        self.run_id = run_id
        self.goals = list(goals or [])
        os.makedirs(self.run_dir, exist_ok=True)

        self.generations_dir = os.path.join(self.run_dir, "generations")
        os.makedirs(self.generations_dir, exist_ok=True)

        self._gen_buffer = {}
        self._gen_written = set()
        self._last_generation = None

        self._closed = False
        atexit.register(self.close)

        self.eval_id = 0
        self.csv_path = os.path.join(self.run_dir, "evaluations.csv")
        fieldnames = [
            "eval_id",
            "run_id",
            "generation",
            "ind_idx",
            "problem_id",
            "algo_id",
            "decision_id",
            "x_internal_hash",
            "decision_json",
            "status",
            "sim_message",
            "g1",
            "feasible",
            "cv",
            "eval_tier",
            "eval_time_sec",
            "eval_cost",
            "total",
        ]

        for g in self.goals:
            fieldnames.append(f"{g}_raw")
        for g in self.goals:
            fieldnames.append(f"{g}_min")

        num_obj = len(self.goals)
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_raw")
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_min")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        try:
            with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)
        except Exception:
            pass

        meta = {
            "problem_id": self.problem_id,
            "algo_id": self.algo_id,
            "run_id": self.run_id,
            "seed": int(self.seed),
            "seed.master": (config_data.get("seed", {}) or {}).get("master", int(self.seed)),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            with open(os.path.join(self.run_dir, "run_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _coerce_generation(self, generation):
        try:
            g = int(generation)
        except Exception:
            g = 0
        if g < 0:
            g = 0
        return g

    def _flush_generation(self, generation: int):
        if generation is None:
            return
        generation = self._coerce_generation(generation)
        if generation in self._gen_written:
            return
        rows = self._gen_buffer.get(generation) or []
        if not rows:
            return

        out_path = os.path.join(self.generations_dir, f"gen_{generation:04d}_pop.csv")

        num_obj = len(self.goals)
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

        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    out_row = {
                        "generation": generation,
                        "ind_idx": r.get("ind_idx"),
                        "parent_ids": "",
                        "rank": "",
                        "crowding": "",
                        "apd": "",
                        "selected_flag": "",
                        "decision_id": r.get("decision_id"),
                        "eval_id": r.get("eval_id"),
                        "cv": r.get("cv"),
                        "feasible": r.get("feasible"),
                    }
                    for i in range(num_obj):
                        out_row[f"f{i+1}_raw"] = r.get(f"f{i+1}_raw")
                        out_row[f"f{i+1}_min"] = r.get(f"f{i+1}_min")
                    writer.writerow(out_row)
        except Exception:
            return

        self._gen_written.add(generation)
        try:
            del self._gen_buffer[generation]
        except Exception:
            pass

    def _decision_id(self, item):
        value = getattr(item, "value", "")
        return hashlib.md5(value.encode("utf-8")).hexdigest()

    def _export_pareto_front(self):
        try:
            if not os.path.exists(self.csv_path):
                return
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return

        if not rows:
            return

        num_obj = len(self.goals)
        if num_obj <= 0:
            return

        cand_rows = []
        F = []
        for row in rows:
            try:
                feasible_val = int(float(row.get("feasible", "0")))
            except Exception:
                feasible_val = 0
            if feasible_val != 1:
                continue

            f_vec = []
            ok = True
            for i in range(num_obj):
                key = f"f{i+1}_min"
                v = row.get(key)
                if v is None or v == "":
                    ok = False
                    break
                try:
                    f_vec.append(float(v))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            cand_rows.append(row)
            F.append(f_vec)

        out_dir = os.path.join(self.run_dir, "final")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pareto_front.csv")
        fieldnames = [
            "decision_id",
            "decision_json",
            "status",
            "sim_message",
            "cv",
            "feasible",
            "g1",
        ]
        for g in self.goals:
            fieldnames.append(f"{g}_raw")
        for g in self.goals:
            fieldnames.append(f"{g}_min")
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_raw")
        for i in range(num_obj):
            fieldnames.append(f"f{i+1}_min")

        if not cand_rows:
            try:
                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            except Exception:
                return
            return

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
        if not out_rows:
            return

        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in out_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception:
            return

    def log_batch(self, items, generation, total_time_sec, tier="true"):
        if not items:
            return

        gen_int = self._coerce_generation(generation)
        if self._last_generation is None:
            self._last_generation = gen_int
        elif gen_int != self._last_generation:
            self._flush_generation(self._last_generation)
            self._last_generation = gen_int

        per_eval_time = 0.0
        if total_time_sec is not None and len(items) > 0:
            per_eval_time = float(total_time_sec) / float(len(items))

        for idx, item in enumerate(items):
            record = getattr(item, "eval_record", None)
            if record is not None:
                props = getattr(record, "objectives_raw", None)
                if not isinstance(props, dict):
                    props = {}

                scores = getattr(record, "objectives_min", None)
                if scores is not None:
                    try:
                        scores = list(scores)
                    except Exception:
                        scores = None

                status_str = getattr(record, "status", None)
                if status_str is None or status_str == "":
                    status_str = "ok"
                else:
                    status_str = str(status_str)

                sim_message = getattr(record, "sim_message", "")
                if sim_message is None:
                    sim_message = ""
                else:
                    sim_message = str(sim_message)

                cv_val = getattr(record, "cv", 0.0)
                try:
                    cv_val = float(cv_val)
                except Exception:
                    cv_val = 0.0
                if cv_val < 0.0:
                    cv_val = 0.0

                g1_val = getattr(record, "g1", None)
                if g1_val is None:
                    g1_val = cv_val
                try:
                    g1_val = float(g1_val)
                except Exception:
                    g1_val = cv_val
                if g1_val < 0.0:
                    g1_val = 0.0

                feasible_int = getattr(record, "feasible", 0)
                try:
                    feasible_int = int(float(feasible_int))
                except Exception:
                    feasible_int = 0
            else:
                props = getattr(item, "property", {}) or {}
                scores = getattr(item, "scores", None)
                constraints = getattr(item, "constraints", None)
                if not isinstance(constraints, dict):
                    constraints = {}

                status_val = constraints.get("status")
                if status_val is None or status_val == "":
                    status_str = "ok"
                else:
                    status_str = str(status_val)

                sim_message_val = constraints.get("sim_message")
                if sim_message_val is None:
                    sim_message = ""
                else:
                    sim_message = str(sim_message_val)

                cv_val = constraints.get("cv")
                if cv_val is None:
                    cv_val = 0.0
                try:
                    cv_val = float(cv_val)
                except Exception:
                    cv_val = 0.0
                if cv_val < 0.0:
                    cv_val = 0.0

                g1_val = constraints.get("g1")
                if g1_val is None:
                    g1_val = cv_val
                try:
                    g1_val = float(g1_val)
                except Exception:
                    g1_val = cv_val
                if g1_val < 0.0:
                    g1_val = 0.0

                feasible_int = constraints.get("feasible")
                if feasible_int is None:
                    feasible_int = int((status_str == "ok") and (cv_val <= 0.0))
                else:
                    try:
                        feasible_int = int(float(feasible_int))
                    except Exception:
                        feasible_int = 1 if bool(feasible_int) else 0

            decision_id = self._decision_id(item)

            total_val = getattr(item, "total", None)
            if total_val is None:
                total_str = ""
            else:
                try:
                    total_str = float(total_val)
                except Exception:
                    total_str = str(total_val)

            row = {
                "eval_id": self.eval_id + 1,
                "run_id": self.run_id,
                "generation": gen_int,
                "ind_idx": idx,
                "problem_id": self.problem_id,
                "algo_id": self.algo_id,
                "decision_id": decision_id,
                "x_internal_hash": decision_id,
                "decision_json": getattr(item, "value", ""),
                "status": status_str,
                "sim_message": str(sim_message),
                "g1": g1_val,
                "feasible": int(feasible_int),
                "cv": cv_val,
                "eval_tier": tier,
                "eval_time_sec": per_eval_time,
                "eval_cost": 1.0,
                "total": total_str,
            }

            for i, g in enumerate(self.goals):
                raw_val = props.get(g)
                row[f"{g}_raw"] = raw_val
                if scores is not None and i < len(scores):
                    row[f"{g}_min"] = scores[i]

                row[f"f{i+1}_raw"] = raw_val
                if scores is not None and i < len(scores):
                    row[f"f{i+1}_min"] = scores[i]

            self.writer.writerow(row)
            self.eval_id += 1

            try:
                self._gen_buffer.setdefault(gen_int, []).append(row)
            except Exception:
                pass

        try:
            self.csv_file.flush()
        except Exception:
            pass

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        try:
            if getattr(self, "csv_file", None) is not None:
                try:
                    self.csv_file.flush()
                except Exception:
                    pass
                self.csv_file.close()
        except Exception:
            pass

        try:
            self._export_pareto_front()
        except Exception:
            pass

        try:
            if getattr(self, "_last_generation", None) is not None:
                self._flush_generation(self._last_generation)
            for g in sorted(list(getattr(self, "_gen_buffer", {}).keys())):
                self._flush_generation(g)
        except Exception:
            pass

    def __del__(self):
        self.close()
