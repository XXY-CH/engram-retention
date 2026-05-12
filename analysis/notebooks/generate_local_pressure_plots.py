#!/usr/bin/env python
"""Generate paper figures from local small-scale pressure-test CSV files."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "local_pressure_20260508"
FIGURE_DIR = ROOT / "analysis" / "figures" / "local_pressure_20260508"
PAPER_DIR = ROOT / "docs" / "proofs"

TASKS = [
    ("needle_256_branchopt", "needle", "Needle-256"),
    ("xor_final_128_branchopt", "xor_final", "XOR-final-128"),
    ("alien_static_128_branchopt", "alien_static", "Alien-static-128"),
]
VARIANTS = [
    ("ours_snapshot_logits", "Ours"),
    ("retnet", "RetNet"),
    ("transformer", "Transformer"),
]
COLORS = {
    "ours_snapshot_logits": "#1f77b4",
    "retnet": "#d62728",
    "transformer": "#2ca02c",
}


def read_rows(result_dir: str, task: str) -> list[dict[str, str]]:
    path = RESULT_ROOT / result_dir / f"{task}_all.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def rows_for(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["variant"] == variant]
    return sorted(selected, key=lambda row: int(row["step"]))


def final_metric(rows: list[dict[str, str]], variant: str, metric: str) -> float:
    selected = rows_for(rows, variant)
    if not selected:
        return float("nan")
    value = selected[-1].get(metric, "")
    return float(value) if value else float("nan")


def save_and_copy(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    analysis_path = FIGURE_DIR / filename
    paper_path = PAPER_DIR / filename
    fig.savefig(analysis_path, bbox_inches="tight")
    shutil.copyfile(analysis_path, paper_path)
    plt.close(fig)


def plot_eval_loss(all_rows: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
    for ax, (_, task, title) in zip(axes, TASKS, strict=True):
        rows = all_rows[task]
        for variant, label in VARIANTS:
            selected = rows_for(rows, variant)
            steps = [int(row["step"]) for row in selected]
            losses = [float(row["eval_loss"]) for row in selected]
            ax.plot(steps, losses, marker="o", linewidth=1.8, color=COLORS[variant], label=label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval loss")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Local pressure-test evaluation loss", y=1.04)
    save_and_copy(fig, "fig_local_pressure_eval_loss.pdf")


def plot_eval_em(all_rows: dict[str, list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (_, task, title) in zip(axes, TASKS, strict=True):
        rows = all_rows[task]
        for variant, label in VARIANTS:
            selected = rows_for(rows, variant)
            steps = [int(row["step"]) for row in selected]
            em = [float(row["eval_exact_match"]) for row in selected]
            ax.plot(steps, em, marker="o", linewidth=1.8, color=COLORS[variant], label=label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Eval exact match")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Local pressure-test exact match", y=1.04)
    save_and_copy(fig, "fig_local_pressure_eval_em.pdf")


def plot_heatmap(
    all_rows: dict[str, list[dict[str, str]]],
    metric: str,
    filename: str,
    title: str,
    fmt: str,
    cmap: str,
) -> None:
    matrix = np.zeros((len(TASKS), len(VARIANTS)), dtype=float)
    for task_idx, (_, task, _) in enumerate(TASKS):
        rows = all_rows[task]
        for variant_idx, (variant, _) in enumerate(VARIANTS):
            matrix[task_idx, variant_idx] = final_metric(rows, variant, metric)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[label for _, label in VARIANTS],
        yticklabels=[title for _, _, title in TASKS],
        cbar_kws={"label": metric.replace("_", " ")},
    )
    ax.set_title(title)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Task")
    save_and_copy(fig, filename)


def write_summary(all_rows: dict[str, list[dict[str, str]]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = FIGURE_DIR / "local_pressure_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "variant", "step", "eval_loss", "eval_exact_match"],
        )
        writer.writeheader()
        for _, task, _ in TASKS:
            for variant, _ in VARIANTS:
                selected = rows_for(all_rows[task], variant)
                final = selected[-1]
                writer.writerow(
                    {
                        "task": task,
                        "variant": variant,
                        "step": final["step"],
                        "eval_loss": final["eval_loss"],
                        "eval_exact_match": final["eval_exact_match"],
                    }
                )


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"
    all_rows = {task: read_rows(result_dir, task) for result_dir, task, _ in TASKS}
    plot_eval_loss(all_rows)
    plot_eval_em(all_rows)
    plot_heatmap(
        all_rows,
        "eval_exact_match",
        "fig_local_pressure_em_heatmap.pdf",
        "Final eval exact match after 120 steps",
        ".3f",
        "YlGnBu",
    )
    plot_heatmap(
        all_rows,
        "eval_loss",
        "fig_local_pressure_loss_heatmap.pdf",
        "Final eval loss after 120 steps",
        ".2f",
        "rocket_r",
    )
    write_summary(all_rows)
    print(f"Wrote local pressure figures to {FIGURE_DIR} and copied PDFs to {PAPER_DIR}")


if __name__ == "__main__":
    main()
