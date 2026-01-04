import shutil
from pathlib import Path


def reset_vmec_inputs() -> None:
    """Reset all vmecpp/calculations* input.w7x files from the master input.w7x.

    Master file:
        problem/stellarator_vmec/input.w7x

    Targets (if they exist):
        problem/stellarator_vmec/vmecpp/calculations*/input.w7x
    """
    base_dir = Path(__file__).resolve().parent
    master_input = base_dir / "input.w7x"

    if not master_input.is_file():
        raise FileNotFoundError(f"Master input.w7x not found: {master_input}")

    vmecpp_dir = base_dir / "vmecpp"
    if not vmecpp_dir.is_dir():
        raise FileNotFoundError(f"VMEC project directory not found: {vmecpp_dir}")

    # Find all calculations* directories under vmecpp
    target_dirs = [d for d in vmecpp_dir.glob("calculations*") if d.is_dir()]

    if not target_dirs:
        print("No calculations* directories found under", vmecpp_dir)
        return

    print(f"Master input: {master_input}")
    for d in target_dirs:
        target_input = d / "input.w7x"
        if not target_input.parent.is_dir():
            continue
        shutil.copy2(master_input, target_input)
        print(f"Copied to: {target_input}")

    print("All VMEC input.w7x files have been reset.")


if __name__ == "__main__":
    reset_vmec_inputs()
