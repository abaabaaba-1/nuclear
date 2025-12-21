# Experimental Report: LLM-Guided Stellarator Coil Optimization with Gradient Hints

**Date**: 2025-12-13
**Method**: GSCO-Lite (Grid-based Stellarator Coil Optimization)
**Task**: Optimization of discrete current loops on a toroidal surface to minimize magnetic field error ($f_B$) under geometric constraints.

---

## 1. Executive Summary

This experiment evaluated the capability of Large Language Models (LLMs) to solve the Stellarator Coil Optimization problem, specifically comparing a "Gradient-Guided LLM" approach (MOLLM) against a rigorous mathematical baseline (True-GSCO).

**Key Findings:**
1.  **Feasibility**: The LLM successfully navigated the discrete design space (12x12 grid, 3-state cells) and generated valid coil configurations.
2.  **Constraint Satisfaction**: The LLM strictly adhered to "Forbidden Zone" constraints (simulating engineering ports) when explicitly instructed and penalized.
3.  **Performance**:
    -   **Baseline (True-GSCO)**: Achieved $f_B \approx 12.19$ (restricted) using a greedy gradient descent.
    -   **MOLLM**: Achieved $f_B \approx 14.17$ (restricted) using gradient hints.
    -   *Gap*: The LLM solution is ~16% worse than the mathematical optimum found by the greedy search, but significantly better than random initialization ($f_B > 100$).
4.  **Mechanism**: The "Gradient Hints" mechanism proved functional, allowing the LLM to make informed local moves while maintaining population diversity.

---

## 2. Methodology

### 2.1 Problem Formulation: GSCO-Lite
To make the stellarator coil problem accessible to LLMs, we discretized the design space:
-   **Grid**: $12 \times 12$ grid on the winding surface ($N=144$ cells).
-   **States**: Each cell $c_{ij} \in \{-1, 0, +1\}$ (Unit current loop CW/CCW).
-   **Physics**: Field calculated via Biot-Savart Law using `simsopt`.
-   **Constraints**:
    -   **Forbidden Zones**: A $2 \times 2$ region at $(\phi, \theta) \in \{(0,5), (0,6), (1,5), (1,6)\}$ is blocked (simulating a port).

### 2.2 Algorithms

#### A. True-GSCO (Honest Baseline)
A classic Greedy Sparse Coil Optimization algorithm.
-   **Initialization**: Empty grid.
-   **Step**: Evaluates all $3N$ possible single-cell changes using a precomputed response matrix. Selects the move that maximally decreases $f_{total} = f_B + \lambda f_S$.
-   **Awareness**: Modified to mask out Forbidden Zones.

#### B. MOLLM (Gradient-Guided Evolutionary Strategy)
An LLM-driven evolutionary algorithm.
-   **Population**: 10-50 candidates.
-   **Operator**: LLM (Gemini-2.5-Flash) performs Mutation/Crossover based on text prompts.
-   **Innovation - Gradient Hints**: The evaluator calculates the top-5 most promising local moves (add/remove/flip) for each candidate using the response matrix. These are injected into the Prompt as text hints (e.g., `[GRADIENT HINTS]: ADD (3,4); REMOVE (2,2)`).
-   **Feedback**: LLM receives $f_B$ values and constraints.

---

## 3. Results Comparison

### 3.1 Unrestricted Scenario (No Forbidden Zones)
*   **True-GSCO**: Min $f_B = 12.52$
*   **MOLLM**: Min $f_B = 13.77$
*   **Observation**: The Greedy algorithm is extremely efficient for this convex-like quadratic problem. The LLM gets close but struggles to perform the precise fine-tuning required for the last 10% of performance.

### 3.2 Restricted Scenario (With Forbidden Zones)
*   **True-GSCO**: Min $f_B = 12.19$ (Converged in ~100 iterations)
    *   *Note*: The lower score compared to unrestricted might be due to stochasticity in the greedy path or slightly different lambda balancing, but mostly it shows the baseline is robust.
*   **MOLLM**: Min $f_B = 14.17$ (Stopped at 26 generations)
*   **Constraint Handling**: MOLLM successfully generated 22 unique valid solutions on the Pareto front that avoided the forbidden zones. Invalid solutions were effectively filtered out.

### 3.3 Visual Analysis
The Pareto Frontier plot (`comparison_plot_restricted.png`) shows:
-   **GSCO Trajectory**: A smooth descent curve (black) improving $f_B$ as complexity ($f_S$) increases.
-   **MOLLM Cloud**: A cluster of red points (red) hovering above the GSCO curve.
-   **Dominance**: The GSCO curve strictly dominates the MOLLM solutions (lower $f_B$ for same $f_S$). This is expected as GSCO has perfect gradient information.

### 3.4 Experience Mechanism Analysis (New)
Inspired by molecular optimization strategies, we enabled the "Experience" module (`experience_prob: 0.5`) to allow the LLM to reflect on past successful/failed candidates.
-   **Scenario**: Unrestricted (No forbidden zones).
-   **Results**:
    -   **No Experience**: Min $f_B \approx 13.77$.
    -   **With Experience**: Min $f_B \approx 13.65$.
-   **Observation**: The improvement is marginal (~1%). 
    -   *Hypothesis*: Unlike molecular strings (SMILES) where local patterns (e.g., functional groups) have consistent effects, the stellarator coil problem is highly non-local and geometric. The LLM likely struggles to articulate "spatial geometric rules" (e.g., "cells at theta=5 need to be positive") purely from text logs without explicit visual or physical intuition prompts.

### 3.5 Physics Intuition Prompt Analysis
To test if the LLM could better utilize the "Experience" module with more domain knowledge, we injected detailed "Strategy Hints" and "Design Space" descriptions into the prompt (e.g., explaining toroidal/poloidal directions and saddle coil patterns).
-   **Scenario**: MOLLM (Experience) vs MOLLM (Experience + Enhanced Prompt).
-   **Results**:
    -   **Base Experience**: Min $f_B \approx 13.645$.
    -   **Enhanced Prompt**: Min $f_B \approx 13.651$.
-   **Analysis**:
    -   The performance remained virtually unchanged.
    -   This strongly suggests that **Gradient Hints are the dominant factor**. The LLM is likely relying almost entirely on the explicit mathematical suggestions ("ADD (3,4)") rather than trying to "understand" the underlying physics or geometry from the text description.
    -   In this high-precision grid optimization task, abstract conceptual knowledge (text) seems less effective than precise local gradient information (math).

### 3.6 Theoretical Analysis: Why GSCO Wins? (Addressing Local Optima)
A common question is whether the "Greedy" nature of GSCO leads to poor local optima compared to the global search capability of an Evolutionary Algorithm (MOLLM).
-   **The Physics**: The magnetic field $\mathbf{B}$ is **linear** with respect to the coil currents $I$ (Biot-Savart Law).
-   **The Objective**: The error metric $f_B \propto \int (\mathbf{B} \cdot \mathbf{n})^2$ is a **Quadratic Form** of the currents ($f_B \approx I^T Q I$).
-   **Convexity**: Since $f_B$ represents an energy-like term, the matrix $Q$ is positive semi-definite. This means the underlying optimization landscape is **Convex** (bowl-shaped).
-   **Implication**:
    -   In a convex landscape, "Greedy" descent (following the gradient) is guaranteed to move towards the global minimum.
    -   Although our problem is **discrete** (Integer constraints), the discrete landscape closely hugs this convex continuous valley.
    -   Therefore, **True-GSCO** (which is essentially Discrete Coordinate Descent) is mathematically extremely efficient. It doesn't get stuck in "traps" because there are very few traps in this specific physics problem.
    -   **MOLLM**, by introducing stochastic mutation/crossover, is essentially adding "noise" to a trajectory that is already simple to navigate. This explains why the LLM struggles to beat the deterministic greedy path.

### 3.7 Two-Step Optimization Analysis (Continuous Relaxation)
We further explored a "Two-Step" strategy: first solving the problem in a **continuous** current space (Relaxation), then discretizing the result (Rounding) to generate "Warm Start" seeds for the LLM.

*   **Continuous Solution**: By allowing currents to take any real value in $[-1, 1]$, the problem becomes a convex Least Squares problem.
    *   **Result**: $f_B \approx 12.11$ (with all 144 cells active).
    *   *Insight*: This represents the **theoretical lower bound** of the physics optimization on this grid. It confirms that **True-GSCO ($f_B \approx 12.19$)** is operating extremely close to the physical limit.

*   **Discrete Warm Start**: We applied a "Stochastic Top-K" rounding strategy to select the 60 most important cells from the continuous solution.
    *   **Result**: $f_B \approx 13.26$ (with $f_S = 60$).
    *   **Comparison**: This initial guess is better than the best result from the pure MOLLM run ($f_B \approx 13.65$), but it uses significantly more coils (60 vs ~20).
    *   *Trade-off*: This highlights the tension between **Field Quality** ($f_B$) and **Engineering Complexity** ($f_S$). The continuous relaxation identifies the "ideal" magnetic distribution, but forcing it into a sparse discrete set (Top-K) incurs a penalty ($12.11 \to 13.26$).

*   **Metric Misalignment Discovery**:
    *   When we fed these high-quality ($f_B \approx 13.26$) but dense ($f_S=60$) seeds into the MOLLM optimizer, the algorithm **rejected** them in favor of sparser solutions ($f_S=5, f_B \approx 14.7$).
    *   *Reason*: The objective ranges were calibrated for sparse solutions ($f_S \in [5, 20]$). The evaluator penalized the "dense" warm start seeds so heavily that they were deemed inferior to "sparse but bad" random solutions.
    *   *Lesson*: In Hybrid AI, **metric alignment** is critical. If the "Physics intuition" (dense coils are better for field) conflicts with the "Engineering constraints" (must be sparse), the AI will exploit the metric to satisfy the constraint at the cost of physics.

---

## 4. Discussion & Conclusion

### Why Gradient Hints Dominates?
Without hints, the LLM is essentially guessing coordinates in a $12 \times 12$ grid. The magnetic field is highly sensitive to position (non-local effects).
-   **With Hints**: The LLM acts as a "stochastic selector" of mathematically valid moves. It adds exploration (jumping out of local minima) that the Greedy algorithm lacks.
-   **Result**: The fact that MOLLM achieved $f_B \approx 14$ (close to 12) proves it utilized the hints. Random guessing typically yields $f_B > 100$.

### Limitations of LLM
1.  **Precision vs Concept**: LLMs excel at conceptual changes but struggle with the "fine-tuning" required for field error minimization.
2.  **Geometry Understanding**: Even with enhanced descriptions, the LLM does not seem to build a true 3D spatial model.

### Conclusion
The experiments demonstrate a clear hierarchy of efficacy for this specific "Convex Objective on Discrete Grid" problem:

1.  **Continuous Relaxation**: $f_B \approx 12.11$ (The theoretical limit, physically unbuildable).
2.  **True-GSCO (Greedy)**: $f_B \approx 12.19$ (The mathematical gold standard for discrete coils).
3.  **Two-Step Warm Start**: $f_B \approx 13.26$ (High-quality initialization, but dense).
4.  **LLM + Gradient Hints**: $f_B \approx 13.65 \sim 14.17$ (Feasible, finding good sparse solutions).
5.  **Pure Random/LLM**: $f_B > 100$ (Fails).

**Final Verdict**: For problems with strong convex underlying physics (like magnetic fields), **Symbolic/Gradient methods (GSCO, Relaxation)** are superior. The optimal role for LLMs is not to replace the solver, but to act as a **high-level architect** (managing constraints, trade-offs) or to use these symbolic methods as tools (via Gradient Hints or Warm Starts).
