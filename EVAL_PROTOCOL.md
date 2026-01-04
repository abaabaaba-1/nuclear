# Semantic-K-RVEA Evaluation Protocol (Phase 0)

> Version: 0.1 (Draft)
> Scope: VMEC / Stellarator Coil / Stellarator Coil GSCO-Lite problems in this repository

This document standardizes **optimization problems, evaluation metrics, experiment protocol, and result formats** for the Semantic-K-RVEA project. It targets **Phase 0**: make all baselines run under a unified protocol with reproducible, parseable outputs.

The design is aligned with three nested loops in Semantic-K-RVEA:

- **Micro-Loop**: K-RVEA + Surrogate + LLM variable operators
- **Macro-Loop**: LLM agent adjusting reference vectors / preference directions
- **Meso-Loop**: LLM agent managing surrogates and evaluation budgets

This protocol defines the *contract* between optimization algorithms, problem evaluators, and logging/analysis tools.

---

## 1. Problem Families & Scope

We currently support three expensive multi-objective problems:

1. **Stellarator VMEC** (`problem/stellarator_vmec`)
   - Decision variables: VMEC Fourier coefficients (continuous)
   - Objectives: volume, aspect_ratio, magnetic_shear
   - Evaluation core: VMEC++

2. **Stellarator Coil (cell-based, GSCO-Lite)** (`problem/stellarator_coil_gsco_lite`)
   - Decision variables: discrete cell grid states on the winding surface (e.g. 12×12 cells)
   - Objectives: f_B (magnetic field error), f_S (sparsity / number of active cells), I_max (maximum current)
   - Evaluation core: Simsopt

3. **(Placeholder) Stellarator Coil (loop-based)** (`problem/stellarator_coil`)
   - Not fully specified here; should follow the same abstractions when added.

This protocol does **not** specify optimization algorithms (NSGA-II, K-RVEA, etc.) themselves, only how they should:

- see the decision space
- receive objective vectors & constraint status
- respect evaluation budgets
- log results in a consistent, parseable way

---

## 2. Unified Problem Abstractions

We introduce conceptual (not necessarily concrete-class) abstractions used across problems:

### 2.1 ProblemSpec

Each problem is described by a `ProblemSpec`:

- `problem_id: str`
  - Examples: `"stellarator_vmec"`, `"stellarator_coil_gsco_lite"`.
- `decision_space: DecisionSpaceSpec`
- `objectives: List[ObjectiveSpec]`
- `constraints: List[ConstraintSpec]`
- `cost_model: CostModelSpec`
- `eval_fn: Callable[[x_internal], EvalOutput]`
  - Provided by the problem module (e.g. VMEC RewardingSystem, GSCO-Lite evaluator).

> **Note**: In this repo, VMEC and GSCO-Lite already implement rich evaluator classes (`RewardingSystem`, `SimpleGSCOEvaluator`). These will be wrapped into or mapped onto `eval_fn` and `ProblemSpec` when the unified evaluator is implemented.

### 2.2 DecisionSpaceSpec

Describes how decision variables are represented internally and semantically.

Common fields:

- `type: "continuous" | "integer" | "mixed" | "grid_binary" | "json_encoded"`
- `dim: int` (for continuous / integer / mixed)
- Optional shape / meta for grid or JSON.

**VMEC (stellarator_vmec)**

- `type = "json_encoded"` (LLM currently manipulates JSON with `new_coefficients`)
- Internally: a JSON string or dict with key→value pairs, e.g.
  - `{ "new_coefficients": {"RBC(0,0)": 5.5, ... } }`
- A semantic representation `decision_json` can be stored for LLM-based operators.

**GSCO-Lite (stellarator_coil_gsco_lite)**

- `type = "grid_binary"`
- `grid_shape = [wf_nPhi, wf_nTheta]` (from `coil_design.wf_nPhi`, `coil_design.wf_nTheta`)
- Internal encoding:
  - LLM and baselines use JSON strings of the form:
    - `{ "cells": [[phi, theta, state], ...] }`, where `state ∈ {-1, 0, +1}`.

### 2.3 ObjectiveSpec

Each objective is described by:

- `name: str`
  - Example: `"volume"`, `"aspect_ratio"`, `"magnetic_shear"`, `"f_B"`, `"f_S"`, `"I_max"`.
- `direction: "min" | "max"`
  - Meaning in **physical space** (before any transformation).
- `transform: "as_is" | "negate" | "normalized" | ...`
  - How to convert from raw physical value to an internal *minimization* target.
- `scale_hint: float` (optional)
  - For normalization and reference-vector scaling (Micro-Loop / K-RVEA).

In current configs:

- `problem/stellarator_vmec/config.yaml`
  - `goals: [volume, aspect_ratio, magnetic_shear]`
  - `optimization_direction: [max, min, max]`

- `problem/stellarator_coil_gsco_lite/config.yaml`
  - `goals: [f_B, f_S, I_max]`
  - `optimization_direction: [min, min, min]`

### 2.4 ConstraintSpec

All constraints are expressed as inequality constraints:

- `g_j(x) ≤ 0` for each `j`.

A `ConstraintSpec` includes:

- `name: str`
- `type: "ineq"` (reserved for future `"eq"`)
- `penalty_weight: float` or leave algorithm-side to interpret.

In practice, many physics constraints are captured as:

- VMEC convergence residuals (`fsqr`, `fsqz`, `fsql` vs `ftolv`)
- Mercier stability (`min_mercier ≥ 0`)
- Geometric validity / forbidden cells in GSCO-Lite.

These should be mapped to explicit `g_j` values when logging (see §4 and §7).

### 2.5 CostModelSpec

Captures per-evaluation cost and tier:

- `tier: "true" | "surrogate" | "cheap_analytic"`
- `per_eval_cost: float` (relative unit cost, e.g. `1.0` for real physics, `0.01` for surrogate).

This is used for:

- Hard budgets (max true evaluations)
- Soft budgets / cost-aware selection in Meso-Loop.

---

## 3. Objective Direction & Internal Minimization Convention

To simplify algorithm implementation (especially K-RVEA and NSGA-II variants), **all optimization algorithms see a vector of minimization objectives**:

- `f_min[i]` is always to be minimized.
- Physical objectives may be maximization or minimization; the conversion is handled at the evaluator/logging layer.

### 3.1 Direction Mapping

Given physical raw objective `f_raw[i]` and direction `dir[i]` in config:

- If `dir[i] == "min"`:
  - `f_min[i] = f_raw[i]`
- If `dir[i] == "max"`:
  - `f_min[i] = -f_raw[i]`

Optionally, a normalization step can transform `f_min` further (e.g. using `objective_ranges`), but the sign convention remains.

### 3.2 Logging Both Raw and Minimization Values

For each objective index `k`:

- `f{k}_raw`: physical value (e.g. volume in m³, magnetic_shear as a scalar difference).
- `f{k}_min`: value actually passed to the optimizer for minimization.

This allows:

- Algorithms (Micro-Loop) to work solely on `f*_min`.
- Analysis scripts and paper figures to use `f*_raw` directly (no confusion about sign).

> Implementation note: VMEC and GSCO-Lite evaluators currently produce various normalized scores. When integrating this protocol, these scores should be clearly separated from `f_raw` and `f_min` in logs.

---

## 4. Constraints & Failure Handling

### 4.1 Unified Constraint Representation

All constraints are represented as:

- `g1, g2, ..., gJ` where each `g_j(x) ≤ 0` for feasibility.

We also derive aggregate indicators:

- `cv = sum_j max(0, g_j)`  (total constraint violation)
- `feasible = int(cv == 0 and status == "ok")`

These are recorded for each evaluation (§7).

### 4.2 Status Codes

Each evaluation returns a `status` string indicating high-level outcome:

- `"ok"`
  - Simulation succeeded; objectives and constraints are reliable.
- `"sim_fail"`
  - Physics solver failed (e.g. VMEC++ crash or non-convergence beyond allowed tolerance).
- `"invalid_input"`
  - Input is clearly invalid before simulation (e.g. forbidden cells used, illegal parameter values).
- `"internal_error"`
  - Unexpected exception in evaluator code.

A human-readable `sim_message` (short error description) should accompany failure statuses.

### 4.3 Penalty Strategy for Failures

To keep algorithms robust, evaluations must always return *some* objective values, even when failures occur.

Recommended behavior:

- For `status != "ok"`:
  - Assign **dominated** or worst-case objectives.
    - E.g. for minimization objectives, set them to a large penalty value; for maximization, set them to a very small or negative large value before conversion.
  - Set `cv` to a very large value (e.g. `1e6`) and `feasible = 0`.

This ensures:

- Populations remain the same size.
- Failed designs are systematically dominated and removed from Pareto fronts.

> VMEC RewardingSystem already applies penalties and default worst values internally. When aligning with this protocol, we will map those to explicit `status`, `g_j`, `cv`, and `feasible` fields in logs.

---

## 5. Evaluation Budget & Cost Layers

Budgets are expressed at two levels: **true expensive evaluations** and optional **surrogate / cheap evaluations**.

### 5.1 Config-Level Budget Fields

In each problem config (e.g. VMEC / GSCO-Lite):

- Under `optimization`:
  - `pop_size: int`
  - `eval_budget: int`
    - Total number of **true expensive evaluations** allowed.
  - `log_freq: int` (optional)
    - Logging frequency per generation.

In the general (future) config structure:

- `budget.max_true_evals: int` (alias of `optimization.eval_budget`)
- `budget.max_surrogate_evals: int` (optional, 0 by default)
- `budget.max_generations: int` (optional upper bound)
- `budget.max_walltime_sec: float` (optional, recorded but not necessarily enforced by all algorithms).

### 5.2 Per-Evaluation Cost Tracking

For each evaluation, logs should include (§7):

- `eval_tier: "true" | "surrogate" | "cheap_analytic"`
- `eval_time_sec: float`
- `eval_cost: float`
  - Typically `1.0` for true expensive physics, smaller for surrogates.

Accumulated `sum(eval_cost)` provides a **soft budget** for Meso-Loop control.

---

## 6. Random Seeds & Reproducibility

To ensure reproducibility across algorithms and problems, we define a consistent seed strategy.

### 6.1 Config Fields

At minimum, configs should contain:

- `seed.master: int`

Optionally, derived seeds:

- `seed.algorithm`
- `seed.problem`
- `seed.llm`
- `seed.surrogate`

If omitted, these should be deterministically derived from `seed.master` (e.g. `master + 1`, `master + 2`, ...).

### 6.2 Runtime Behavior

At the beginning of each run:

- Set global RNGs:
  - `numpy.random.seed(...)`
  - `random.seed(...)`
  - `torch.manual_seed(...)` (if used)
- Ensure problem modules (VMEC / GSCO-Lite) and optimizers use either:
  - Passed-in RNG instances, or
  - The globally-set RNGs only.

### 6.3 Logging Seeds

Each run directory (see §7) must contain a `run_meta.json` with:

- `seed.master`
- Derived seeds (if any)
- Timestamp
- Git commit hash (if available)
- `problem_id`, `algo_id`

The full `config.yaml` for the run should be copied into the run directory as well.

---

## 7. Output Directory Structure & Logging Schema

### 7.1 Directory Layout

For any run (baseline or Semantic-K-RVEA variant), results should be placed under:

```text
results/{problem_id}/{algorithm_id}/{run_id}/
  config.yaml         # full config used for this run
  run_meta.json       # seeds, git hash, timestamps, problem+algo ids
  evaluations.csv     # one row per evaluation (true/surrogate)
  generations/
    gen_0000_pop.csv
    gen_0001_pop.csv
    ...
  final/
    pareto_front.csv  # final non-dominated set (in physical objective space)
    archive.csv       # external archive, if used by the algorithm
  logs/               # optional: algorithm-specific logs
```

Notes:

- `run_id` can be a combination of experiment name, seed, and timestamp, e.g. `"exp1_seed42_2025-12-22T10-30-00"`.
- For backwards compatibility, existing save paths (e.g. `moo_results/`) may be symlinked or mapped into this structure.

### 7.2 evaluations.csv Schema

Each row corresponds to **one evaluation** (true or surrogate) of one decision.

Required columns (minimal draft):

- Run identifiers:
  - `eval_id: int` (global counter within the run)
  - `run_id: str`
  - `generation: int` (may be `-1` or `NaN` for non-generational algorithms)
  - `ind_idx: int` (index within its generation, if applicable)
  - `problem_id: str`
  - `algo_id: str`

- Decision info:
  - `decision_id: str` or int (hash or index for the unique decision)
  - `x_internal_hash: str` (hash of internal representation)
  - `decision_json: str` (optional; JSON-encoded semantic decision).

- Status & constraints:
  - `status: str` (see §4.2)
  - `sim_message: str`
  - `g1, g2, ..., gJ: float`
  - `cv: float`
  - `feasible: int`

- Objectives:
  - `f1_raw, f2_raw, ..., fM_raw`
  - `f1_min, f2_min, ..., fM_min`

- Cost & timing:
  - `eval_tier: str`
  - `eval_time_sec: float`
  - `eval_cost: float`

### 7.3 gen_XXXX_pop.csv Schema

Each file records the **population snapshot** of one generation (or iteration), from the algorithm’s viewpoint.

Suggested columns:

- `generation: int`
- `ind_idx: int`
- `parent_ids: str` (comma-separated parent decision_ids)
- `rank: int` (Pareto front rank)
- `crowding: float` (NSGA-II)
- `apd: float` (K-RVEA angular penalty distance, if applicable)
- `selected_flag: int` (1 if selected for the next generation)
- `decision_id: str`
- `eval_id: int` (link back to evaluations.csv)
- `f*_raw`, `f*_min`, `cv`, `feasible` (duplicated for convenience)

This snapshot format is primarily for:

- Macro-Loop (LLM-Reference agent) to inspect population distribution.
- Offline analysis (HV / IGD over generations).

### 7.4 final/pareto_front.csv

The final front should be exported in **physical objective space**:

- `decision_id`
- `f*_raw`
- `f*_min`
- `cv`, `feasible`
- Optional semantic info (e.g. `decision_json`).

---

## 8. Config Conventions & Examples

This section aligns the protocol with existing configs.

### 8.1 Common Top-Level Fields

All optimization configs should follow these baseline keys:

- `exper_name: str`
- `description: str`
- `save_dir: str`
- `save_suffix: str`
- `resume: bool`

- `model:` (when LLM is used)
  - `name: str`
  - `base_url: str`
  - `api_key: str`
  - `prompt_module: str`
  - `experience_prob, crossover_prob, mutation_prob, explore_prob: float`

- `goals: [obj1, obj2, ...]`
- `optimization_direction: [min|max, ...]`

- `prompt_info_path: str` (LLM prompt config)
- `evalutor_path: str` (Python module path to evaluator, e.g. `problem.stellarator_vmec.evaluator`).

- `objective_ranges: { obj_name: [min, max], ... }`
- `llm_constraints: { ... }` (problem-specific constraints for LLM operators).

- `optimization:`
  - `pop_size: int`
  - `eval_budget: int`
  - `log_freq: int`

Optional (for Phase 0 but recommended):

- `seed.master: int`

### 8.2 VMEC Example (Existing)

From `problem/stellarator_vmec/config.yaml`:

- Goals & directions:
  - `goals: [volume, aspect_ratio, magnetic_shear]`
  - `optimization_direction: [max, min, max]`
- VMEC-specific:
  - `vmec.project_path`
  - `vmec.input_file`
  - `vmec.output_file`
  - `vmec.save_wout_mode`
  - `vmec.save_wout_top_k`
- Budget:
  - `optimization.pop_size: 100`
  - `optimization.eval_budget: 5000`
- Objective ranges for normalization:
  - `objective_ranges.volume: [26.0, 29.5]`
  - `objective_ranges.aspect_ratio: [10.5, 11.5]`
  - `objective_ranges.magnetic_shear: [0.9, 1.0]`

These should be used both for:

- Internal normalization (
  - e.g. mapping to [0,1] scores).
- Logging as `f*_raw` + `f*_min` in evaluation records.

### 8.3 GSCO-Lite Example (Existing)

From `problem/stellarator_coil_gsco_lite/config.yaml`:

- Goals & directions:
  - `goals: [f_B, f_S, I_max]`
  - `optimization_direction: [min, min, min]`
- Plasma boundary:
  - `plasma_boundary.wout_file`
  - `plasma_boundary.plas_n`
- Coil design:
  - `coil_design.wf_nPhi`, `coil_design.wf_nTheta`
  - `coil_design.unit_current`
  - `coil_design.winding_surface_expansion`
  - `coil_design.use_background_field`
  - `coil_design.eval_workers`
- Budget:
  - `optimization.pop_size: 50`
  - `optimization.eval_budget: 5000`
  - `optimization.num_offspring`
  - `optimization.log_freq`
  - `optimization.parallel_offspring`
  - `optimization.initial_population_file`
- Objective ranges:
  - `objective_ranges.f_B: [12.3, 16.0]`
  - `objective_ranges.f_S: [0, 60]`
  - `objective_ranges.I_max: [0.2, 0.4]`

These fields should be kept, and used for:

- Normalization in `SimpleGSCOEvaluator`.
- Logging `f*_raw` + `f*_min`.

---

## 9. Minimal Runnable Example (Phase 0 Target)

Phase 0 requires at least one **small-budget optimization run** that:

- Uses this unified protocol (or a close approximation).
- Writes results to a structured directory with `config.yaml`, `run_meta.json`, and at least one of:
  - `evaluations.csv`
  - `gen_*.csv` or equivalent population snapshots.

### 9.1 Suggested Example: GSCO-Lite Baseline

Starting point: `run_gsco_baselines.py` with `StandardGA` or `RandomSearch`.

- Config: `problem/stellarator_coil_gsco_lite/config.yaml`
- Temporarily set a **small budget** in the config for Phase 0 tests, e.g.:
  - `optimization.pop_size: 10`
  - `optimization.eval_budget: 40`
- Run one of:
  - `RandomSearch`
  - `StandardGA`
  - `SimulatedAnnealing`
  - `GreedyGSCO`

Integration tasks (to be implemented later):

1. Wrap `SimpleGSCOEvaluator.evaluate` with a small adapter that:
   - Measures wall-clock time per evaluation.
   - Assigns `status`, `sim_message`, constraints, and `f*_raw`, `f*_min` following this protocol.
   - Appends rows to `evaluations.csv`.

2. In `BaselineOptimizer` (and derivatives), add per-generation logging:
   - Export population metrics into `generations/gen_XXXX_pop.csv`.

3. Store `config.yaml` and a generated `run_meta.json` under:

```text
results/stellarator_coil_gsco_lite/{algo_name}/{run_id}/
```

Once this minimal example is in place, **any other baseline or Semantic-K-RVEA variant** should reuse the same evaluator wrapper and logger, differing only in how they generate and select candidates.

---

## 10. Relation to Semantic-K-RVEA Layers

### 10.1 Micro-Loop

- Uses `DecisionSpaceSpec` / `ObjectiveSpec` to:
  - Work in a consistent *minimization* objective space (`f_min`).
  - Apply reference vectors and APD without worrying about direction flips.
- May access `decision_json` for semantic LLM-based variation.
- Relies on `evaluations.csv` to compute performance indicators if needed.

### 10.2 Macro-Loop

- Inspects `generations/gen_*.csv` and `evaluations.csv` to:
  - Understand current front distribution.
  - Monitor constraint satisfaction and failure rates.
  - Adjust reference vectors or preference directions based on these statistics.

### 10.3 Meso-Loop

- Uses `eval_tier`, `eval_cost`, `eval_time_sec` and budget configuration to:
  - Decide when to call a true expensive evaluator vs a surrogate.
  - Control the mix of high/low fidelity evaluations over time.

---

## 11. Future Work & Open Points (Beyond Phase 0)

Phase 0 focuses on **specification and minimal adoption**. Further steps (future phases) include:

- Implementing a shared `Evaluator` and `EvalLogger` utility module in the codebase.
- Refactoring existing evaluators (VMEC / GSCO-Lite) to emit standardized `EvalRecord` objects mapping directly to `evaluations.csv` rows.
- Extending configs with explicit `seed.master` and run metadata.
- Adding analysis scripts that read `evaluations.csv` / `gen_*.csv` and compute HV/IGD, etc.

These are *not strictly part of Phase 0 spec*, but the current document is written to make them natural follow-ups.
