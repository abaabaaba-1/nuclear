from __future__ import annotations

from typing import Any


def get_evaluation(evaluate_metric: Any, smiles: Any):
    try:
        from problem.molecules.evaluator import get_evaluation as _get_evaluation
    except Exception as e:
        raise RuntimeError(
            "get_evaluation is only available when the molecules evaluator is installed/usable."
        ) from e

    return _get_evaluation(evaluate_metric, smiles)


def eval_mo_results(dataset: Any, obj: Any, ops: Any = None):
    return []


def mean_sr(r: Any):
    return 0.0, 0.0, 0.0
