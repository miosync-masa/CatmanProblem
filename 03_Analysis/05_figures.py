from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

from analysis_utils import FIGURES_DIR, JUDGMENT_ITEMS, RESULTS_DIR, ensure_dirs


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.04, "Conceptual decomposition", "Separate evidence, observer estimates,\nqueries, and normative targets"),
        (0.365, "Interventional\nthought experiment", "Perturb one proposed support while\nholding the comparison structure fixed"),
        (0.69, "Empirical diagnosis", "Test whether judgments are unitary,\ncue-specific, or query-specific"),
    ]
    colors = ["#D9EAF7", "#FBE5C8", "#DCEFE7"]
    for (x, title, subtitle), color in zip(boxes, colors):
        box = FancyBboxPatch(
            (x, 0.43),
            0.27,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.3,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.135, 0.66, title, ha="center", va="center", weight="bold", linespacing=1.0)
        ax.text(x + 0.135, 0.535, subtitle, ha="center", va="center", fontsize=8.3)
    for x0, x1 in [(0.31, 0.365), (0.635, 0.69)]:
        ax.annotate("", xy=(x1, 0.60), xytext=(x0, 0.60), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(
        0.5,
        0.23,
        r"$E_k \;\longrightarrow\; \hat{C}^{(O)}_k \;\longrightarrow\; J^{(O)}_q$",
        ha="center",
        va="center",
        fontsize=17,
    )
    ax.text(
        0.5,
        0.07,
        "The Cat-Man Problem is a diagnostic form of inquiry, not a theory of personal identity.",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=9,
    )
    save_figure(fig, "fig0_catman_framework")


def manipulation_heatmap() -> None:
    data = pd.read_csv(RESULTS_DIR / "manipulation_selectivity.csv")
    contrast_order = ["Body", "Memory", "Function", "Self-ID uncertain", "Transition", "Self-denial anchor"]
    item_order = ["pc_sub", "pc_info", "pc_func", "pc_self", "pc_tran"]
    labels = ["Body", "Memory", "Function", "Self", "Transition"]
    matrix = (
        data.pivot(index="contrast", columns="item", values="mean_difference")
        .reindex(index=contrast_order, columns=item_order)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(7.9, 5.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(contrast_order)), contrast_order)
    ax.set_xlabel("Perceived continuity dimension")
    ax.set_ylabel("Intervention contrast vs baseline")
    ax.set_title("Manipulations selectively changed their targeted continuity estimates")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) > 3.2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Mean within-participant difference (condition − C01)")
    ax.text(
        0,
        -0.17,
        "N = 50 manipulation-arm participants. Negative values indicate reduced perceived continuity.",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=8,
    )
    save_figure(fig, "fig1_manipulation_selectivity")


def judgment_profiles() -> None:
    data = pd.read_csv(RESULTS_DIR / "judgment_effects.csv")
    contrasts = ["Body", "Memory", "Function", "Self-ID uncertain", "Transition", "Self-denial anchor"]
    titles = ["Body", "Memory", "Function", "Self-ID uncertain", "Transition", "Self-denial"]
    labels = ["Identity", "Likeness", "Relationship", "Obligation", "Treatment"]
    colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY, RED]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), sharex=True, sharey=True)
    for ax, contrast, title, color in zip(axes.flat, contrasts, titles, colors):
        sub = data[data["contrast"].eq(contrast)].set_index("item").reindex(JUDGMENT_ITEMS)
        x = np.arange(len(JUDGMENT_ITEMS))
        y = sub["mean_difference"].to_numpy()
        low = y - sub["ci95_low"].to_numpy()
        high = sub["ci95_high"].to_numpy() - y
        ax.errorbar(x, y, yerr=[low, high], color=color, marker="o", lw=1.8, capsize=3)
        ax.axhline(0, color="#999999", lw=0.9, ls="--")
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(-3.8, 0.8)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    axes[0, 0].set_ylabel("Mean difference vs C01")
    axes[1, 0].set_ylabel("Mean difference vs C01")
    fig.suptitle("The same intervention produced different judgment profiles across queries", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Points are within-participant mean differences; bars are 95% CIs. N = 49 after attention exclusion.",
        ha="center",
        color=GRAY,
        fontsize=8.5,
    )
    fig.tight_layout()
    save_figure(fig, "fig2_judgment_profiles")


def deny_parallel_figure() -> None:
    effects = pd.read_csv(RESULTS_DIR / "judgment_effects.csv")
    deny = effects[effects["contrast"].eq("Self-denial anchor")].set_index("item").reindex(JUDGMENT_ITEMS)
    parallel = pd.read_csv(RESULTS_DIR / "parallel_patterns_deidentified.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    ax = axes[0]
    y = np.arange(len(JUDGMENT_ITEMS))
    means = deny["mean_difference"].to_numpy()
    xerr = np.vstack(
        [means - deny["ci95_low"].to_numpy(), deny["ci95_high"].to_numpy() - means]
    )
    ax.errorbar(means, y, xerr=xerr, fmt="o", color=RED, capsize=3, lw=1.8)
    ax.axvline(0, color="#999999", ls="--", lw=0.9)
    ax.set_yticks(y, ["Identity", "Likeness", "Relationship", "Obligation", "Treatment"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean difference: A09 − C01")
    ax.set_title("A. Self-denial affected queries unequally")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7)

    ax = axes[1]
    rng = np.random.default_rng(20260814)
    jitter_x = rng.normal(0, 0.055, len(parallel))
    jitter_y = rng.normal(0, 0.055, len(parallel))
    scatter = ax.scatter(
        parallel["j_ab"] + jitter_x,
        parallel["j_ac"] + jitter_y,
        c=parallel["j_bc"],
        cmap="viridis",
        norm=Normalize(1, 7),
        s=42,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.axvline(4.5, color="#777777", ls="--", lw=0.9)
    ax.axhline(4.5, color="#777777", ls="--", lw=0.9)
    ax.set_xlim(0.7, 7.3)
    ax.set_ylim(0.7, 7.3)
    ax.set_xticks(range(1, 8))
    ax.set_yticks(range(1, 8))
    ax.set_xlabel("Pre-X = surviving human (j_ab)")
    ax.set_ylabel("Pre-X = generated cat (j_ac)")
    ax.set_title("B. Parallel anchor judgments")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Human = cat rating (j_bc)")
    count_both = int(parallel["both_predecessor_links_high"].sum())
    count_discordant = int(parallel["transitivity_like_discordance"].sum())
    ax.text(
        0.02,
        0.98,
        f"Both predecessor links high: {count_both}/49\nWith BC not high: {count_discordant}/49",
        transform=ax.transAxes,
        va="top",
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    save_figure(fig, "fig3_deny_and_parallel")


def main() -> None:
    ensure_dirs()
    framework_figure()
    manipulation_heatmap()
    judgment_profiles()
    deny_parallel_figure()


if __name__ == "__main__":
    main()
