#!/usr/bin/env python
"""Generate eval-loss and eval-EM curves for repaired rerun-old diagnostics."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "rerun_old_after_fix_20260508"
FIGURE_DIR = ROOT / "analysis" / "figures" / "rerun_old_after_fix_20260508"
PAPER_DIR = ROOT / "docs" / "proofs"

TASKS = [
    ("needle_256", "needle", "Needle-256"),
    ("xor_final_128", "xor_final", "XOR-final-128"),
    ("alien_static_128", "alien_static", "Alien-static-128"),
]
VARIANTS = [
    ("ours_snapshot_logits", "MESA"),
    ("retnet", "RetNet"),
    ("transformer", "Transformer"),
]
COLORS = {
    "ours_snapshot_logits": "#1f77b4",
    "retnet": "#d62728",
    "transformer": "#2ca02c",
}


def read_rows(result_dir: str, task: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / result_dir / f"{task}_all.csv").open() as handle:
        return list(csv.DictReader(handle))


def rows_for(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["variant"] == variant]
    return sorted(selected, key=lambda row: int(row["step"]))


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    analysis_path = FIGURE_DIR / filename
    paper_path = PAPER_DIR / filename
    fig.savefig(analysis_path, bbox_inches="tight")
    shutil.copyfile(analysis_path, paper_path)
    plt.close(fig)


def plot_metric(metric: str, ylabel: str, filename: str, sharey: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.25), sharey=sharey)
    for ax, (result_dir, task, title) in zip(axes, TASKS, strict=True):
        rows = read_rows(result_dir, task)
        for variant, label in VARIANTS:
            selected = rows_for(rows, variant)
            steps = [int(row["step"]) for row in selected]
            values = [float(row[metric]) for row in selected]
            ax.plot(steps, values, marker="o", linewidth=1.8, color=COLORS[variant], label=label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if metric == "eval_exact_match":
            ax.set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Repaired rerun diagnostics: {ylabel}", y=1.04)
    save(fig, filename)


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"
    plot_metric("eval_loss", "Eval loss", "fig_rerun_eval_loss.pdf", sharey=False)
    plot_metric("eval_exact_match", "Eval exact match", "fig_rerun_eval_em.pdf", sharey=True)
    print(f"Wrote repaired rerun plots to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
