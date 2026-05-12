#!/usr/bin/env python
"""Measure small CPU/GPU runtime and memory proxies for model variants."""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.train_synthetic import build_model, make_batch  # noqa: E402


def make_train_namespace(args: argparse.Namespace, variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        task="needle",
        variants=variant,
        steps=1,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=192,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=0.0,
        learning_rate=3e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        use_branch_optimizer=True,
        per_group_grad_clip=True,
        guard_lr_mult=5.0,
        cache_lr_mult=3.0,
        guard_beta1=0.0,
        cache_beta1=0.5,
        seed=42,
        device=args.device,
        log_interval=20,
        eval_batches=0,
        eval_seed=10042,
        eval_split="train",
        eval_drop_modules=[],
        out_dir="",
        engram_layer=2,
        engram_slots=args.engram_slots,
        engram_max_ngram=3,
        engram_hash_heads=4,
        engram_table_device=args.engram_table_device,
        attnres_every=4,
        attnres_max_sources=8,
        attnres_distance_penalty=0.0,
        branch_init_scale=1e-4,
        use_milestones=True,
        milestone_ttl=256,
        milestone_gamma=0.999,
        use_snapshots=True,
        max_snapshots=8,
        use_snapshot_logit_bias=True,
        snapshot_logit_scale=1.0,
        retention_gamma=0.95,
        needle_password_len=1,
        alien_num_pairs=4,
        alien_static_key_count=32,
        alien_static_value_count=32,
        alien_static_train_keys=24,
    )


def rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB. This repo is run on macOS,
    # but keep the fallback conservative.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


@torch.no_grad()
def probe_variant(args: argparse.Namespace, variant: str) -> dict[str, float | str | int]:
    train_args = make_train_namespace(args, variant)
    device = torch.device(args.device)
    model = build_model(train_args, variant, train_args.vocab_size).to(device)
    batch = make_batch(train_args, device, split="train")
    engram = next((layer.engram for layer in getattr(model, "layers", []) if layer.engram is not None), None)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        _ = model(batch.input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rss_before = rss_mb()
    start = time.perf_counter()
    for _ in range(args.iters):
        _ = model(batch.input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    rss_after = rss_mb()

    row: dict[str, float | str | int] = {
        "variant": variant,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "iters": args.iters,
        "ms_per_forward": elapsed * 1000 / args.iters,
        "rss_delta_mb": max(0.0, rss_after - rss_before),
        "rss_peak_mb": rss_after,
        "engram_table_device": args.engram_table_device or "model_device",
        "engram_table_devices_actual": ",".join(sorted(str(d) for d in engram.table_devices()))
        if engram is not None
        else "",
        "engram_table_mb": engram.table_memory_bytes() / (1024 * 1024)
        if engram is not None
        else 0.0,
    }
    if device.type == "cuda":
        row["cuda_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="ours_snapshot_logits,retnet,transformer")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--engram-slots", type=int, default=8192)
    parser.add_argument(
        "--engram-table-device",
        default=None,
        help="Optional Engram table device, e.g. cpu for host-memory offload.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="experiments/results/ordered_pressure_20260508/07_runtime_probe/runtime.csv",
    )
    args = parser.parse_args()

    rows = [
        probe_variant(args, variant.strip())
        for variant in args.variants.split(",")
        if variant.strip()
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote runtime probe to {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
