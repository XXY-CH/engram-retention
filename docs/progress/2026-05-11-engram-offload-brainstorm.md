# Engram Offload Brainstorm

Date: 2026-05-11

Status: captured from the architecture/proof cleanup session.

## Core Position

Engram should be treated as an offloadable static or semi-static conditional
memory table, not as a generic document retrieval system and not as proof that
the backbone has learned all knowledge internally.

The practical split is:

```text
RetNet backbone: streaming state and ordinary token processing
TokenCopyBuffer / Snapshot: rare high-entropy exact-copy paths
Engram: frequent local patterns, static facts, canonicalized lookup keys
Block AttnRes: depth-wise reuse of intermediate representations
```

## Innovation Pool

| Idea | Innovation | Shortcut |
|---|---|---|
| Hot/cold Engram cache | Keep frequent slots on accelerator and cold slots in host/NVMe storage. | Avoid full-table accelerator residency. |
| Engram hit-rate telemetry | Log hash hit distribution, collision proxies, gate values, and module-drop deltas. | Use mechanism metrics before expensive benchmark scaling. |
| Canonicalizer before Engram | Normalize aliases, formatting, and synonyms into stable lookup keys. | Keep Engram as hard hash lookup instead of solving semantic search inside the table. |
| Oracle-to-learned snapshot | Use oracle capture to establish an upper bound, then distill a capture policy. | Avoid learning capture and readout simultaneously at first. |
| Margin ledger | Track capture margin, attention margin, collision margin, and logit margin. | Each proof only needs to prove a margin transfer. |
| Session Engram | Store repeated user/session facts in temporary namespace tables. | Avoid repeatedly uploading the same long context. |
| Static knowledge shards | Split course/API/domain facts into replaceable namespaces. | Update knowledge by replacing a table shard, not retraining the backbone. |
| Engram as anti-KV-cache | Move high-frequency local-pattern memory out of the sequence cache. | Reduce pressure on RetNet/Transformer context mechanisms. |
| CopyBuffer + Engram split | Route exact raw tokens to CopyBuffer and canonicalized facts to Engram. | Keeps exact-copy and knowledge-lookup proofs separate. |
| Deferred MoE | Reintroduce MoE only after memory paths have separate evidence. | Avoid mixing router collapse with memory-path debugging. |

## Near-Term Experiments

1. Compare default Engram table placement with `engram_table_device="cpu"` on the
   same runtime probe.
2. Add a hot/cold cache prototype after the CPU offload hook is stable.
3. Record table size, table device, forward latency, and module-drop accuracy
   together so memory savings are not separated from model quality.

## Guardrail

Do not claim that offload proves knowledge correctness. Offload only changes
where the table lives. Correctness still requires key hit rate, bounded
collision noise, gate safety, and downstream margin.

## 2026-05-11 Offload Probe

Experiment command family:

```text
python experiments/probe_runtime.py --variants ours_snapshot_logits --device mps --seq-len 256 --batch-size 8 --d-model 48 --n-layers 4 --engram-slots 65536 --warmup 2 --iters 20
```

Output files are under `experiments/results/engram_offload_probe_20260511/`,
which is ignored by `.gitignore`.

| Placement | Actual table device | Engram table | Forward latency |
|---|---:|---:|---:|
| model device | `mps:0` | 144.0 MB | 106.40 ms |
| CPU offload | `cpu` | 144.0 MB | 111.81 ms |

Interpretation:

```text
CPU offload removes the 144 MB Engram table from accelerator residency in this
probe, with about 5.1% forward-latency overhead.
```

This is a promising systems trade-off for large static tables. It is still a
synchronous CPU lookup path, not a hot/cold cache or async prefetch engine.
