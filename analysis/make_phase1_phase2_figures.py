import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_phase1_baseline_suite_overview(out_path: str) -> None:
    # Matrix-style overview: rows=problems, cols=baseline families
    problems = ["VMEC", "GSCO-Lite"]
    baselines = [
        "Random",
        "GA",
        "NSGA-II",
        "SMSEMOA",
        "MOEA/D",
        "RVEA",
        "K-RVEA",
        "MOLLM (legacy)",
    ]

    # Coverage in this repo (based on run_all_baselines.py / run_gsco_baselines.py)
    coverage = {
        "VMEC": {
            "Random": False,
            "GA": True,
            "NSGA-II": True,
            "SMSEMOA": True,
            "MOEA/D": True,
            "RVEA": True,
            "K-RVEA": True,
            "MOLLM (legacy)": True,
        },
        "GSCO-Lite": {
            "Random": True,
            "GA": True,
            "NSGA-II": True,
            "SMSEMOA": True,
            "MOEA/D": True,
            "RVEA": True,
            "K-RVEA": False,
            "MOLLM (legacy)": True,
        },
    }

    data = []
    for p in problems:
        row = [1.0 if coverage[p].get(b, False) else 0.0 for b in baselines]
        data.append(row)

    fig_w = max(10.0, 1.2 * len(baselines))
    fig_h = 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(data, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)

    ax.set_yticks(list(range(len(problems))))
    ax.set_yticklabels(problems)

    ax.set_xticks(list(range(len(baselines))))
    ax.set_xticklabels(baselines, rotation=25, ha="right")

    # Annotate cells
    for i, p in enumerate(problems):
        for j, b in enumerate(baselines):
            ok = bool(coverage[p].get(b, False))
            ax.text(j, i, "✓" if ok else "—", ha="center", va="center", fontsize=14, color="#1b1b1b")

    ax.set_title("Phase1 Baseline Suite Overview (repo coverage)")

    # Legend as text
    ax.text(
        1.01,
        0.5,
        "✓ implemented\n— not available",
        transform=ax.transAxes,
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_phase2_fusionopt_v1_overview(out_path: str) -> None:
    # A clean loop diagram with minimal crossings.
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axis_off()

    def box(xy: Tuple[float, float], w: float, h: float, text: str, fc: str) -> patches.FancyBboxPatch:
        x, y = xy
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            facecolor=fc,
            edgecolor="#2f3b52",
            linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, wrap=True)
        return rect

    def v_arrow(x: float, y0: float, y1: float):
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#2f3b52"),
        )

    ax.text(0.5, 0.97, "Phase2 FusionOpt v1 Main Loop (clean overview)", ha="center", va="top", fontsize=16)

    # Vertical layout (centered)
    x = 0.18
    w = 0.64
    h = 0.085
    ys = [0.84, 0.71, 0.58, 0.45, 0.32, 0.19]

    box((x, ys[0]), w, h, "1) Init population (generate_initial_population)", fc="#f5f7ff")
    box((x, ys[1]), w, h, "2) Operator sampling (crossover / mutation / resample / ...)", fc="#f5f7ff")
    box((x, ys[2]), w, h, "3) Gate + Dedup (JSON validate + canonicalize)", fc="#f5f7ff")
    box((x, ys[3]), w, h, "4) HeuRepair (optional domain repair)", fc="#fff7e6")
    box((x, ys[4]), w, h, "5) Expensive evaluate (RewardingSystem.evaluate)", fc="#e8fff0")
    box((x, ys[5]), w, h, "6) Selection + Archive + Protocol logging (NSGA-II + EvalLogger)", fc="#eef2ff")

    # Straight arrows
    xc = x + w / 2
    for i in range(len(ys) - 1):
        v_arrow(xc, ys[i], ys[i + 1] + h)

    # Loop arrow from step 6 back to step 2
    loop = patches.FancyArrowPatch(
        (x + w, ys[5] + h / 2),
        (x + w, ys[1] + h / 2),
        connectionstyle="arc3,rad=0.35",
        arrowstyle="->",
        linewidth=1.8,
        color="#2f3b52",
        mutation_scale=14,
    )
    ax.add_patch(loop)
    ax.text(x + w + 0.03, (ys[5] + ys[1]) / 2 + h / 2, "repeat (generations)", rotation=90, va="center", fontsize=10)

    ax.text(
        0.5,
        0.06,
        "Notes: v1 prioritizes stability + reproducibility. Phase4 adds LLM-SemOp (w_llm > 0) and must validate/fallback to avoid sim_fail cascades.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "showcase", "figures")
    _ensure_dir(out_dir)

    out1 = os.path.join(out_dir, "phase1_baseline_suite_overview.png")
    out2 = os.path.join(out_dir, "phase2_fusionopt_v1_overview.png")

    make_phase1_baseline_suite_overview(out1)
    make_phase2_fusionopt_v1_overview(out2)

    print("Wrote:")
    print(f"- {out1}")
    print(f"- {out2}")


if __name__ == "__main__":
    main()
