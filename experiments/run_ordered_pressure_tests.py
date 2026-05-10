#!/usr/bin/env python
"""Run the ordered early-stage pressure tests used by the manuscript.

The runs are intentionally small and CPU-friendly. They are mechanism checks,
not benchmark-grade training jobs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "results" / "ordered_pressure_20260508"

COMMON = [
    "--batch-size",
    "8",
    "--eval-batches",
    "2",
    "--eval-split",
    "train",
    "--d-model",
    "48",
    "--n-layers",
    "4",
    "--d-ff",
    "192",
    "--log-interval",
    "20",
    "--use-milestones",
    "--use-snapshots",
    "--use-snapshot-logit-bias",
    "--use-branch-optimizer",
    "--per-group-grad-clip",
]


def run_train(name: str, args: list[str], dry_run: bool) -> None:
    command = [sys.executable, "experiments/train_synthetic.py", *args]
    print(f"\n=== {name} ===")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def schedule() -> list[tuple[str, list[str]]]:
    runs: list[tuple[str, list[str]]] = []

    runs.append(
        (
            "01_module_drop_needle",
            [
                "--task",
                "needle",
                "--variants",
                "ours_snapshot_logits",
                "--seq-len",
                "256",
                "--needle-password-len",
                "1",
                "--retention-gamma",
                "0.95",
                "--steps",
                "120",
                "--eval-drop-modules",
                "snapshot,engram,attnres",
                "--out-dir",
                str(OUT_ROOT / "01_module_drop_needle"),
                *COMMON,
            ],
        )
    )

    for k in [0, 1, 2, 4, 8]:
        runs.append(
            (
                f"02_k_sweep_k{k}",
                [
                    "--task",
                    "needle",
                    "--variants",
                    "ours_snapshot_logits",
                    "--seq-len",
                    "256",
                    "--needle-password-len",
                    "1",
                    "--retention-gamma",
                    "0.95",
                    "--steps",
                    "80",
                    "--max-snapshots",
                    str(k),
                    "--out-dir",
                    str(OUT_ROOT / f"02_k_sweep_k{k}"),
                    *COMMON,
                ],
            )
        )

    for seq_len in [128, 256, 512, 1024]:
        runs.append(
            (
                f"03_context_sweep_{seq_len}",
                [
                    "--task",
                    "needle",
                    "--variants",
                    "ours_snapshot_logits,retnet,transformer",
                    "--seq-len",
                    str(seq_len),
                    "--needle-password-len",
                    "1",
                    "--retention-gamma",
                    "0.95",
                    "--steps",
                    "80",
                    "--out-dir",
                    str(OUT_ROOT / f"03_context_sweep_{seq_len}"),
                    *COMMON,
                ],
            )
        )

    for gamma in ["0.90", "0.95", "0.98", "0.995"]:
        runs.append(
            (
                f"04_gamma_sweep_{gamma}",
                [
                    "--task",
                    "needle",
                    "--variants",
                    "ours_snapshot_logits,retnet,transformer",
                    "--seq-len",
                    "256",
                    "--needle-password-len",
                    "1",
                    "--retention-gamma",
                    gamma,
                    "--steps",
                    "80",
                    "--out-dir",
                    str(OUT_ROOT / f"04_gamma_sweep_{gamma.replace('.', '')}"),
                    *COMMON,
                ],
            )
        )

    for seed in [42, 43, 44]:
        runs.append(
            (
                f"05_multiseed_{seed}",
                [
                    "--task",
                    "needle",
                    "--variants",
                    "ours_snapshot_logits,retnet,transformer",
                    "--seq-len",
                    "256",
                    "--needle-password-len",
                    "1",
                    "--retention-gamma",
                    "0.95",
                    "--steps",
                    "80",
                    "--seed",
                    str(seed),
                    "--eval-seed",
                    str(10000 + seed),
                    "--out-dir",
                    str(OUT_ROOT / f"05_multiseed_{seed}"),
                    *COMMON,
                ],
            )
        )

    runs.append(
        (
            "06_heldout_alien_keys",
            [
                "--task",
                "alien_static",
                "--variants",
                "ours_snapshot_logits,retnet,transformer",
                "--seq-len",
                "128",
                "--steps",
                "120",
                "--eval-split",
                "test",
                "--out-dir",
                str(OUT_ROOT / "06_heldout_alien_keys"),
                *[item for item in COMMON if item != "--eval-split" and item != "train"],
            ],
        )
    )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Comma-separated run-name prefixes to execute.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_prefixes = [item.strip() for item in args.only.split(",") if item.strip()]
    for name, command_args in schedule():
        if selected_prefixes and not any(name.startswith(prefix) for prefix in selected_prefixes):
            continue
        run_train(name, command_args, args.dry_run)


if __name__ == "__main__":
    main()
