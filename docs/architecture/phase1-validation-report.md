# Phase 1 Validation Report

Date: 2026-05-11

Status: **MECHANISM CHECKPOINT COMPLETE**

## Summary

All three targeted mechanisms have positive synthetic evidence at seq_len up to
1024. The dense small model (~200K parameters, d_model=64) reaches perfect
exact-match on the listed local diagnostics, but these are not broad benchmark
or deployment claims.

## Results

| Capability | Task | seq_len | eval_em | Convergence |
|-----------|------|---------|---------|-------------|
| Long-context recall | needle | 1024 | 1.000 | 400 steps |
| Static fact memory | alien_static | 64 | 1.000 | 400 steps |
| Recursive reasoning | XOR | 1024 | 1.000 | 600 steps |
| Single-step reasoning | xor_final | 128 | 1.000 | 200 steps |

## Architecture Components Validated

| Component | Role | Proof |
|-----------|------|-------|
| RetentionLayer | Base sequence mixing (parallel + recurrent) | RetNet (Sun et al. 2023) |
| HashedNgramEngram | Static key-value lookup | Deterministic hash |
| BlockAttentionResidual | Depth-axis gradient flow | Proof 17 (non-dilution) |
| MilestoneRetentionGate | Time-axis decay protection | Proof 22 |
| MilestoneSnapshotReadout | Hidden state snapshots | - |
| TokenCopyBuffer | Conditional exact-copy logit path | Proof 29 (logit-margin condition) |

## Key Discoveries

1. **RetNet-only variants fail the local needle task** — recurrent state
   compression does not preserve enough recoverable token identity in the
   tested setting. eval_em = 0.000 in the cited runs.

2. **TokenCopyBuffer supplies a direct copy path** — stores raw token embeddings
   and projects through the embedding matrix to logits. eval_em = 0.984 at 128
   in the cited local run.

3. **Positional encoding keys are critical** — without position info in copy attention,
   performance degrades at seq_len > 256 (0.797 at 512). With positional keys:
   eval_em = 1.000 at 512 and 1024 in the cited local runs.

4. **abs() prevents residual-scale sign reversal** — AdamW weight decay can push
   small parameters through zero. abs() preserves the non-negative scale
   assumption used by conditional value-path arguments (Proof 28).

5. **No cross-task regression** — TokenCopyBuffer is neutral-to-helpful on
   alien_static (eval_loss 0.0072 vs 0.0084 without buffer).

## Ablation Results (seq_len=512, needle task)

| Ablation | eval_em | Conclusion |
|----------|---------|------------|
| No TokenCopyBuffer | 0.016 | RetNet alone can't solve needle |
| Buffer, no positional keys | 0.797 | Content-only attention insufficient |
| Buffer + positional keys | 1.000 | Position info supports discrimination under this task distribution |
| d_model=128 (larger) | 0.609 | More capacity doesn't help |
| Branch optimizer (higher LR) | 0.797 | Learning rate isn't the bottleneck |

## Compute Cost

- Model: ~200K parameters, d_model=64, 8 layers
- Device: Apple MPS (M-series GPU)
- Training: 400-600 steps to convergence
- Time: ~2-5 minutes per experiment at seq_len=1024

## Phase 2 Progress: O(1) Recurrent Inference

Status: **RECURRENT MODE SMOKE-VALIDATED**

### Implementation

| Component | Status | Notes |
|-----------|--------|-------|
| `RecurrentState` | Done | Fixed-size dataclass, no growth with seq_len |
| `forward_recurrent_step()` | Done | Single-token O(1) processing through all layers |
| `init_recurrent_state()` | Done | Factory method matching model config |
| Recurrent eval in train_synthetic | Done | `--eval-recurrent` flag for step-by-step eval |

### Recurrent vs Parallel Equivalence

Recurrent eval_em matches parallel eval_em across all tested lengths:

| seq_len | steps to parallel 1.000 | steps to recurrent 1.000 | Max difference |
|---------|------------------------|-------------------------|----------------|
| 128 | 180 | 180 | 0.000 |
| 512 | 200 | 200 | 0.015 |
| 1024 | 200 | 200 | 0.000 |

### Memory Profile

State tensor sizes are constant:
- Retention states: `n_layers × [batch, n_heads, head_dim, head_dim]`
- Copy buffer: `[batch, max_snapshots, d_model]` + valid + pos_ids
- Snapshot buffer: `[batch, max_snapshots, d_model]` + valid
- Total: O(d² × L + d × K) where L=layers, K=max_snapshots — independent of sequence length

### Remaining Phase 2 Work

1. Test at seq_len 1024+ with recurrent mode
2. Chunkwise processing for sequences exceeding max_seq_len position embeddings
3. Profile wall-clock throughput (steps/sec) of recurrent vs parallel
4. Engram disk offload for static knowledge at scale

### Engram Offload Implementation Note

`HashedNgramEngram` now supports an explicit `engram_table_device` config. When
set to `"cpu"`, the static hash embedding tables stay in host memory even if the
rest of the model is moved to an accelerator. Forward lookup hashes on the
input-token device, performs table reads on the table device, and moves only the
retrieved memory activation back to the residual-stream device.

This is a CPU/host-memory offload hook, not a full NVMe paging engine. It
models the intended memory split:

- accelerator memory: RetNet backbone, projection/gate layers, hot activations;
- host memory: large static Engram hash tables;
- future work: hot-slot cache, async prefetch, and disk-backed table shards.
