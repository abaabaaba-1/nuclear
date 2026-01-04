from __future__ import annotations

from typing import Dict, List, Tuple


def ensure_protocol_fields_vmec(item, penalty_cv: float = 1e6) -> None:
    constraints = getattr(item, "constraints", None)
    if not isinstance(constraints, dict):
        constraints = {}

    is_feasible = constraints.get("is_feasible")
    feasible = 1 if float(is_feasible or 0.0) >= 1.0 else 0

    if feasible == 1:
        constraints["status"] = "ok"
        constraints["sim_message"] = ""
        constraints["cv"] = 0.0
        constraints["g1"] = 0.0
        constraints["feasible"] = 1
    else:
        constraints["status"] = "sim_fail"
        constraints["sim_message"] = constraints.get("sim_message") or "infeasible_or_failed"
        constraints["cv"] = float(penalty_cv)
        constraints["g1"] = float(penalty_cv)
        constraints["feasible"] = 0

    item.constraints = constraints


def ensure_protocol_fields_gsco(item, penalty_cv: float = 1e6) -> None:
    constraints = getattr(item, "constraints", None)
    if not isinstance(constraints, dict):
        constraints = {}

    hints = getattr(item, "gradient_hints", None) or []
    violation_msg = ""
    if isinstance(hints, list):
        for h in hints:
            try:
                hs = str(h)
            except Exception:
                continue
            if hs.startswith("VIOLATION:"):
                violation_msg = hs
                break

    is_feasible = constraints.get("is_feasible")
    feasible_raw = constraints.get("feasible")
    if is_feasible is not None:
        try:
            feasible = 1 if float(is_feasible or 0.0) >= 1.0 else 0
        except Exception:
            feasible = 0
    elif feasible_raw is not None:
        try:
            feasible = int(float(feasible_raw))
        except Exception:
            feasible = 1 if bool(feasible_raw) else 0
    else:
        feasible = 1

    if violation_msg:
        constraints["status"] = "invalid_input"
        constraints["sim_message"] = violation_msg
        constraints["cv"] = float(penalty_cv)
        constraints["g1"] = float(penalty_cv)
        constraints["feasible"] = 0
    elif feasible == 1:
        constraints["status"] = "ok"
        constraints["sim_message"] = ""
        constraints["cv"] = 0.0
        constraints["g1"] = 0.0
        constraints["feasible"] = 1
    else:
        sim_message = constraints.get("sim_message")
        constraints["status"] = "constraint_fail"
        constraints["sim_message"] = "" if sim_message is None else str(sim_message)

        cv_val = constraints.get("cv")
        if cv_val is None:
            cv_val = penalty_cv
        try:
            cv_val = float(cv_val)
        except Exception:
            cv_val = float(penalty_cv)
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

        constraints["cv"] = cv_val
        constraints["g1"] = g1_val
        constraints["feasible"] = 0

    item.constraints = constraints


def ensure_protocol_fields(problem_id: str, items: List, penalty_cv: float = 1e6) -> None:
    for it in items:
        if problem_id == "stellarator_vmec":
            ensure_protocol_fields_vmec(it, penalty_cv=penalty_cv)
        elif problem_id == "stellarator_coil_gsco_lite":
            ensure_protocol_fields_gsco(it, penalty_cv=penalty_cv)
        else:
            # fallback
            constraints = getattr(it, "constraints", None)
            if not isinstance(constraints, dict):
                constraints = {}
            if "status" not in constraints:
                constraints["status"] = "ok"
            if "sim_message" not in constraints:
                constraints["sim_message"] = ""
            if "cv" not in constraints:
                constraints["cv"] = 0.0
            if "g1" not in constraints:
                constraints["g1"] = constraints.get("cv", 0.0)
            if "feasible" not in constraints:
                constraints["feasible"] = int((constraints.get("status") == "ok") and float(constraints.get("cv") or 0.0) <= 0.0)
            it.constraints = constraints


def make_penalty_item(item_factory, decision_json: str, goals: List[str], penalty_cv: float = 1e6):
    it = item_factory.create(decision_json)
    # Minimal fields for EvalLogger
    it.property = {g: None for g in goals}
    it.scores = [1.0 for _ in goals]
    it.total = None
    it.constraints = {
        "status": "invalid_input",
        "sim_message": "dropped_by_evaluator_or_gate",
        "cv": float(penalty_cv),
        "g1": float(penalty_cv),
        "feasible": 0,
    }
    return it
