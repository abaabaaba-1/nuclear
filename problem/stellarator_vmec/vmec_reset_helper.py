from __future__ import annotations

"""Utility helpers for automatically resetting VMEC inputs between runs."""

import shutil
from pathlib import Path
from typing import Literal, Optional

try:
    from .reset_vmec_inputs import reset_vmec_inputs
except Exception as exc:  # pragma: no cover - import errors logged at runtime
    reset_vmec_inputs = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_BASE_DIR = Path(__file__).resolve().parent
_MASTER_INPUT = _BASE_DIR / "input.w7x"


def _copy_master_to(project_path: str) -> None:
    target_dir = Path(project_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_input = target_dir / "input.w7x"
    if not _MASTER_INPUT.is_file():
        raise FileNotFoundError(f"Master input.w7x not found: {_MASTER_INPUT}")
    shutil.copy2(_MASTER_INPUT, target_input)


def _reset_all_projects() -> None:
    if reset_vmec_inputs is not None:
        reset_vmec_inputs()
    else:
        raise RuntimeError(f"reset_vmec_inputs unavailable: {_IMPORT_ERROR}")


def maybe_reset_vmec_inputs(evalutor_path: Optional[str], project_path: Optional[str], when: Literal["pre", "post"]) -> None:
    """Reset VMEC inputs if the current config uses the stellarator evaluator."""
    if not evalutor_path or not evalutor_path.startswith("problem.stellarator_vmec"):
        return
    try:
        if project_path:
            _copy_master_to(project_path)
        else:
            _reset_all_projects()
        print(f"[VMEC Reset] Completed ({when}-run) -> {project_path or 'all calculations* directories'}")
    except Exception as exc:  # pragma: no cover
        print(f"[VMEC Reset] Warning ({when}-run): {exc}")
