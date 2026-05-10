#!/usr/bin/env python
"""Train a tiny protocol-compliant model and plot internal mechanism heatmaps."""

from __future__ import annotations

import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from experiments.train_synthetic import (  # noqa: E402
    MARK_THOUGHT,
    QUERY,
    START,
    build_model,
    build_optimizer,
    clip_gradients,
    make_batch,
    masked_lm_loss,
    set_seed,
)
from experiments.probe_runtime import make_train_namespace  # noqa: E402


OUT_DIR = ROOT / "experiments" / "results" / "ordered_pressure_20260508" / "08_internal_heatmaps"
FIGURE_NAME = "fig_internal_mechanism_heatmap.pdf"


def train_model() -> tuple[torch.nn.Module, object]:
    args = make_train_namespace(
        type(
            "RuntimeArgs",
            (),
            {"batch_size": 8, "seq_len": 256, "d_model": 48, "n_layers": 4, "device": "cpu"},
        )(),
        "ours_snapshot_logits",
    )
    args.steps = 120
    args.eval_batches = 0
    set_seed(args.seed)
    random.seed(args.seed)
    model = build_model(args, "ours_snapshot_logits", args.vocab_size)
    optimizer = build_optimizer(args, model)
    model.train()
    for _ in range(args.steps):
        batch = make_batch(args, torch.device("cpu"), split="train")
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.input_ids, return_metrics=True)
        loss = masked_lm_loss(logits, batch.target_ids, batch.loss_mask)
        loss.backward()
        clip_gradients(args, model, optimizer)
        optimizer.step()
    return model, args


def make_probe_input(seq_len: int, vocab_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260508)
    tokens = torch.randint(16, vocab_size, (1, seq_len), generator=generator)
    tokens[:, 0] = START
    for pos, value in [(18, 31), (64, 73), (112, 117)]:
        tokens[:, pos] = value
        tokens[:, pos + 1] = MARK_THOUGHT
    tokens[:, -2] = QUERY
    tokens[:, -1] = 117
    return tokens


def grouped_engram_gate(gate: torch.Tensor, groups: int = 8) -> torch.Tensor:
    # gate: [batch, seq, d]
    mean_gate = gate.mean(dim=0)
    seq_len, d_model = mean_gate.shape
    width = d_model // groups
    trimmed = mean_gate[:, : width * groups]
    return trimmed.view(seq_len, groups, width).mean(dim=-1)


def write_csv_matrix(path: Path, matrix: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for row in matrix.detach().cpu().tolist():
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, args = train_model()
    model.eval()
    probe = make_probe_input(128, args.vocab_size)
    with torch.no_grad():
        _, _, diagnostics = model(probe, return_metrics=True, return_diagnostics=True)

    snapshot = diagnostics["snapshot_weights"][0].detach().cpu()
    attnres_key = next(key for key in diagnostics if key.endswith("_attnres_weights"))
    attnres = diagnostics[attnres_key][0].detach().cpu()
    engram_key = next(key for key in diagnostics if key.endswith("_engram_gate"))
    engram = grouped_engram_gate(diagnostics[engram_key]).detach().cpu()
    milestone = diagnostics["milestone_gate"][0].detach().cpu()

    write_csv_matrix(OUT_DIR / "snapshot_attention.csv", snapshot)
    write_csv_matrix(OUT_DIR / "attnres_depth_attention.csv", attnres)
    write_csv_matrix(OUT_DIR / "engram_gate_groups.csv", engram)
    write_csv_matrix(OUT_DIR / "milestone_gate.csv", milestone)

    sns.set_theme(style="white", context="paper", font_scale=0.9)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    heatmaps = [
        (
            snapshot.T,
            "Snapshot attention over selected token-time slots",
            "Token position",
            "Snapshot slot",
        ),
        (attnres.T, "Block AttnRes depth-source attention", "Token position", "Depth source"),
        (engram.T, "Engram gate grouped by hidden channels", "Token position", "Gate group"),
        (milestone.T, "Milestone retention gate by head", "Token position", "Retention head"),
    ]
    for ax, (matrix, title, xlabel, ylabel) in zip(axes, heatmaps, strict=True):
        sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    figure_path = OUT_DIR / FIGURE_NAME
    fig.savefig(figure_path, bbox_inches="tight")
    shutil.copyfile(figure_path, ROOT / "docs" / "proofs" / FIGURE_NAME)
    plt.close(fig)
    print(f"Wrote internal mechanism heatmap to {figure_path}")


if __name__ == "__main__":
    main()
