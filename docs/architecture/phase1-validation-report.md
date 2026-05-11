# Phase 1 Validation Report

Date: 2026-05-11

Status: **COMPLETE**

## Summary

All three core capabilities verified on synthetic tasks at seq_len up to 1024.
The dense small model (~200K parameters, d_model=64) achieves perfect performance
on recall, memory, and reasoning benchmarks.

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
| TokenCopyBuffer | Exact long-range token recall | Proof 29 (Cauchy-Schwarz) |

## Key Discoveries

1. **RetNet cannot solve needle task alone** — recurrent state compresses token-level
   information below recoverable fidelity. eval_em = 0.000 at all lengths.

2. **TokenCopyBuffer solves needle via embedding projection** — stores raw token
   embeddings, projects through embedding matrix to logits. eval_em = 0.984 at 128.

3. **Positional encoding keys are critical** — without position info in copy attention,
   performance degrades at seq_len > 256 (0.797 at 512). With positional keys:
   eval_em = 1.000 at 512 and 1024.

4. **abs() prevents negative residual scale** — AdamW weight decay pushes small
   parameters through zero. abs() preserves non-dilution bounds (Proof 28).

5. **No cross-task regression** — TokenCopyBuffer is neutral-to-helpful on
   alien_static (eval_loss 0.0072 vs 0.0084 without buffer).

## Ablation Results (seq_len=512, needle task)

| Ablation | eval_em | Conclusion |
|----------|---------|------------|
| No TokenCopyBuffer | 0.016 | RetNet alone can't solve needle |
| Buffer, no positional keys | 0.797 | Content-only attention insufficient |
| Buffer + positional keys | 1.000 | Position info solves discrimination |
| d_model=128 (larger) | 0.609 | More capacity doesn't help |
| Branch optimizer (higher LR) | 0.797 | Learning rate isn't the bottleneck |

## Compute Cost

- Model: ~200K parameters, d_model=64, 8 layers
- Device: Apple MPS (M-series GPU)
- Training: 400-600 steps to convergence
- Time: ~2-5 minutes per experiment at seq_len=1024

## Next: Phase 2

Scale from 1024 to 1M+ context via:
1. Recurrent mode (O(1) per step) for base RetNet
2. Stateful TokenCopyBuffer (incremental collect + readout)
3. Chunkwise processing for long sequences
4. Engram disk offload for static knowledge
