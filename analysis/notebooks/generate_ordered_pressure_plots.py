#!/usr/bin/env python
"""Generate plots from the repaired ordered pressure-test suite."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "ordered_pressure_20260508"
FIGURE_DIR = ROOT / "analysis" / "figures" / "ordered_pressure_20260508"
PAPER_DIR = ROOT / "docs" / "proofs"

VARIANTS = ["ours_snapshot_logits", "retnet", "transformer"]
LABELS = {
    "ours_snapshot_logits": "MESA",
    "retnet": "RetNet",
    "transformer": "Transformer",
}
COLORS = {
    "ours_snapshot_logits": "#1f77b4",
    "retnet": "#d62728",
    "transformer": "#2ca02c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def final_row(path: Path, variant: str) -> dict[str, str]:
    rows = [row for row in read_csv(path) if row["variant"] == variant]
    return sorted(rows, key=lambda row: int(row["step"]))[-1]


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    analysis_path = FIGURE_DIR / filename
    paper_path = PAPER_DIR / filename
    fig.savefig(analysis_path, bbox_inches="tight")
    shutil.copyfile(analysis_path, paper_path)
    plt.close(fig)


def plot_context_sweep() -> list[dict[str, object]]:
    seqs = [128, 256, 512, 1024]
    summary: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for variant in VARIANTS:
        em_values = []
        for seq in seqs:
            row = final_row(RESULT_ROOT / f"03_context_sweep_{seq}" / "needle_all.csv", variant)
            em = float(row["eval_exact_match"])
            loss = float(row["eval_loss"])
            em_values.append(em)
            summary.append(
                {
                    "suite": "context",
                    "setting": seq,
                    "variant": variant,
                    "eval_exact_match": em,
                    "eval_loss": loss,
                }
            )
        ax.plot(
            seqs, em_values, marker="o", linewidth=1.8, color=COLORS[variant], label=LABELS[variant]
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs, labels=[str(seq) for seq in seqs])
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("Needle context length")
    ax.set_ylabel("Eval exact match")
    ax.set_title("Context-length pressure after retention-mask fix")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig_ordered_context_em.pdf")
    return summary


def plot_k_sweep() -> list[dict[str, object]]:
    ks = [0, 1, 2, 4, 8]
    em_values = []
    summary: list[dict[str, object]] = []
    for k in ks:
        row = final_row(RESULT_ROOT / f"02_k_sweep_k{k}" / "needle_all.csv", "ours_snapshot_logits")
        em = float(row["eval_exact_match"])
        loss = float(row["eval_loss"])
        em_values.append(em)
        summary.append(
            {
                "suite": "k_sweep",
                "setting": k,
                "variant": "ours_snapshot_logits",
                "eval_exact_match": em,
                "eval_loss": loss,
            }
        )
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(ks, em_values, marker="o", linewidth=1.9, color=COLORS["ours_snapshot_logits"])
    ax.set_xticks(ks)
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("Snapshot budget K")
    ax.set_ylabel("Eval exact match")
    ax.set_title("Snapshot budget sweep on Needle-256")
    ax.grid(alpha=0.25)
    save(fig, "fig_ordered_k_sweep.pdf")
    return summary


def plot_gamma_sweep() -> list[dict[str, object]]:
    gammas = [("090", 0.90), ("095", 0.95), ("098", 0.98), ("0995", 0.995)]
    summary: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for variant in VARIANTS:
        values = []
        for suffix, gamma in gammas:
            row = final_row(RESULT_ROOT / f"04_gamma_sweep_{suffix}" / "needle_all.csv", variant)
            em = float(row["eval_exact_match"])
            values.append(em)
            summary.append(
                {
                    "suite": "gamma",
                    "setting": gamma,
                    "variant": variant,
                    "eval_exact_match": em,
                    "eval_loss": float(row["eval_loss"]),
                }
            )
        ax.plot(
            [gamma for _, gamma in gammas],
            values,
            marker="o",
            linewidth=1.8,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("Forced RetNet decay gamma")
    ax.set_ylabel("Eval exact match")
    ax.set_title("Gamma sweep on Needle-256")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig_ordered_gamma_sweep.pdf")
    return summary


def plot_multiseed() -> list[dict[str, object]]:
    seeds = [42, 43, 44]
    matrix = np.zeros((len(VARIANTS), len(seeds)))
    summary: list[dict[str, object]] = []
    for row_idx, variant in enumerate(VARIANTS):
        for col_idx, seed in enumerate(seeds):
            row = final_row(RESULT_ROOT / f"05_multiseed_{seed}" / "needle_all.csv", variant)
            em = float(row["eval_exact_match"])
            matrix[row_idx, col_idx] = em
            summary.append(
                {
                    "suite": "multiseed",
                    "setting": seed,
                    "variant": variant,
                    "eval_exact_match": em,
                    "eval_loss": float(row["eval_loss"]),
                }
            )
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[str(seed) for seed in seeds],
        yticklabels=[LABELS[variant] for variant in VARIANTS],
        cbar_kws={"label": "Eval EM"},
    )
    ax.set_title("Needle-256 exact match across seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Variant")
    save(fig, "fig_ordered_seed_summary.pdf")
    return summary


def write_summary(rows: list[dict[str, object]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "ordered_pressure_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["suite", "setting", "variant", "eval_exact_match", "eval_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"
    rows: list[dict[str, object]] = []
    rows.extend(plot_context_sweep())
    rows.extend(plot_k_sweep())
    rows.extend(plot_gamma_sweep())
    rows.extend(plot_multiseed())
    write_summary(rows)
    print(f"Wrote ordered pressure plots to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
