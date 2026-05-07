# Synthetic Snapshot Ablation Results

Created: 2026-05-04

## Question

Does `MilestoneSnapshotReadout` solve only exact copy, or does it also improve
non-copy reasoning / lookup tasks?

## Setup

Common settings:

```text
steps=120
batch_size=16 for xor/alien, 8 for needle
retention_gamma=0.95
milestone_ttl >= sequence length
held-out eval enabled with eval_batches=4
```

Variants:

```text
ours_nosnapshot
ours_snapshot
ours_snapshot_logits
retnet
transformer
```

## Results

| Task | Best Variant | Main Metric | Interpretation |
|---|---:|---:|---|
| Needle 512 high-entropy exact recall | `ours_snapshot_logits` | held-out exact match `0.344` | Positive evidence for explicit snapshot-to-logit value path. |
| XOR recurrence | `ours_snapshot_logits` by loss, `ours_nosnapshot`/`retnet` by accuracy | held-out loss `0.289`, held-out accuracy `0.922/0.930` for non-snapshot baselines | Snapshot logits reduce loss but do not clearly improve token accuracy; sequence exact match is too strict for this task. |
| Alien dictionary | `ours_snapshot` by early EM, `ours_nosnapshot` by final loss | held-out exact match peak `0.109`, final loss `3.704` | No robust snapshot advantage; current alien task does not prove Engram/static memory use yet. |

Detailed CSVs:

```text
experiments/results/synthetic_needle_512_snapshot_ablation_eval/needle_all.csv
experiments/results/synthetic_xor_snapshot_ablation_eval/xor_all.csv
experiments/results/synthetic_alien_snapshot_ablation_eval/alien_all.csv
```

## Theory Adjustment

The data supports only the following limited claim:

```text
MilestoneSnapshotReadout provides a bounded O(B_time) token-time value path for
exact high-entropy recall when capture and readout-margin conditions hold.
```

It does not yet support:

```text
snapshot readout improves general reasoning
snapshot readout improves static/fuzzy knowledge lookup
snapshot readout replaces Engram
snapshot readout replaces depth-axis Block AttnRes
```

## Metric Correction

`exact_match` is appropriate for needle and alien single-answer tasks. It is not
appropriate for the current XOR task because the mask covers many output
positions; whole-sequence exact match stays at zero even when token accuracy is
high. XOR should be judged primarily by held-out loss and token accuracy, or the
task should be rewritten to ask only for the final bit.

## Next Experiment

1. Add a `xor_final` task with one masked answer token.
2. Add an `alien_static` task where the same dictionary is reused across batches,
   so Engram can actually learn reusable hashed facts.
3. Add module-drop eval flags to disable snapshot/Engram/AttnRes at evaluation
   time and measure causal contribution directly.

## Follow-up: Cleaner Tasks Added

Added to `experiments/train_synthetic.py`:

```text
xor_final
alien_static
```

### xor_final

This task masks only one final answer token, so exact match is meaningful.

Results:

| Variant | Best Held-out EM | Final Held-out Loss |
|---|---:|---:|
| `ours_nosnapshot` | `0.578` | `0.804` |
| `ours_snapshot` | `1.000` | `0.067` |
| `ours_snapshot_logits` | `1.000` | `0.000` |
| `retnet` | `0.547` | `0.833` |
| `transformer` | `1.000` | `0.089` |

Interpretation: bounded snapshot readout can provide a very strong single-answer
readout path. This is useful, but it also reinforces the shortcut warning:
snapshot success must be separated from claims about general reasoning.

### alien_static

This task uses a stable key-to-value mapping across batches.

Results:

| Variant | Best Held-out EM | Final Held-out Loss |
|---|---:|---:|
| `ours_nosnapshot` | `1.000` | `0.584` |
| `ours_snapshot` | `1.000` | `0.517` |
| `ours_snapshot_logits` | `0.969` | `0.867` |
| `retnet` | `1.000` | `0.543` |
| `transformer` | `1.000` | `1.518` |

Interpretation: this task is too easy to isolate Engram. RetNet learns the static
mapping without needing Engram. A harder Engram diagnostic must use many more
keys, lower model capacity, or explicit module-drop evaluation.

## Updated Next Experiment

1. Make `alien_static` harder: larger key space, smaller `d_model`, possibly
   train/test key splits.
2. Add a `needle_recall_only` eval where the snapshot source is intentionally
   wrong or disabled, to measure capture dependence directly.
3. Use module-drop eval in every serious run.

## Tooling Added: Module-Drop Evaluation

`experiments/train_synthetic.py` now supports:

```text
--eval-drop-modules snapshot,engram,attnres
```

For RetNet/Engram models, held-out evaluation can temporarily disable one module
at a time and emit fields such as:

```text
eval_no_snapshot_loss
eval_no_snapshot_exact_match
eval_no_engram_loss
eval_no_attnres_loss
```

Smoke output:

```text
experiments/results/synthetic_module_drop_smoke/needle_all.csv
```

This is now the preferred way to distinguish correlation from causal module
contribution.

## Follow-up: Harder alien_static

`alien_static` now supports:

```text
--alien-static-key-count
--alien-static-value-count
--alien-static-train-keys
--eval-split train|test|all
```

The value mapping is now a deterministic hash-style mapping, not the old easy
`value = key + offset` rule.

Hard setting:

```text
key_count=128
value_count=128
train_keys=96
d_model=16
n_layers=2
engram_layer=1
branch_init_scale=0.1
eval_drop_modules=engram
```

Train-key evaluation:

| Variant | Eval EM | Eval Loss | No-Engram EM | No-Engram Loss |
|---|---:|---:|---:|---:|
| `ours_nosnapshot` | `0.992` | `2.009` | `0.969` | `2.222` |
| `retnet` | `0.922` | `2.196` | `0.922` | `2.196` |

Test-key evaluation:

| Variant | Eval EM | Eval Loss | No-Engram EM | No-Engram Loss |
|---|---:|---:|---:|---:|
| `ours_nosnapshot` | `0.000` | `6.480` | `0.000` | `6.472` |
| `retnet` | `0.000` | `6.501` | `0.000` | `6.501` |

Interpretation:

```text
Engram has weak but measurable causal contribution on seen static facts:
  +0.023 EM and -0.213 loss relative to no-Engram eval.

Engram does not generalize to unseen key facts:
  test-key EM remains 0.0.
```

This matches the corrected theorem boundary: hard Engram supports seen/static
factual memory under hash/collision/margin conditions; it is not fuzzy semantic
retrieval and cannot answer facts never stored or trained.
