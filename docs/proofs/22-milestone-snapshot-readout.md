# Milestone Snapshot Readout Theorem

Created: 2026-05-04

Status: conditional A-Time theorem for the optional exact token-time snapshot
readout. This is not Kimi Attention Residuals. It is the bounded sparse-anchor
mechanism needed when exact high-entropy recall cannot be recovered from a
saturated RetNet recurrent state.

## 0. Motivation

The milestone retention gate proof shows how a protected state component avoids
ordinary exponential decay when cumulative leakage is bounded. It does not show
that the original token value remains queryable after many unrelated updates.

For high-entropy exact-copy tasks, the failure mode is:

```text
amplitude survives in RetNet state
but value identity is corrupted by accumulated updates
```

So exact recall needs a separate value path:

```text
h_i -> snapshot cache -> readout at query time -> logits/output
```

## 1. Minimal Model

Let a milestone source at token time `i` produce hidden vector:

```text
s_i = h_i
```

A bounded snapshot cache stores:

```text
C = {(k_j, v_j)}_{j=1}^{B_time}
```

where `B_time` is a fixed small budget. A later query hidden state `q_T` reads:

```text
p_{T,j} = softmax(q_T^T k_j / sqrt(d))_j
z_T = sum_j p_{T,j} v_j
```

The model may inject `z_T` as a residual branch or use it as a logit-bias branch:

```text
h_T' = h_T + lambda_s R z_T
```

or:

```text
logits_T' = logits_T + lambda_s W z_T
```

## 2. Snapshot Capture Condition

Let `i*` be the critical token-time source. The first condition is simple but
unavoidable:

```text
i* in C
```

If the marker is misplaced, the budget evicts the source, or the cache stores an
already-corrupted representation, no readout theorem can recover the value.

The milestone recall probability must therefore be measured separately:

```text
r_capture = Pr(i* in C)
```

## 3. Readout Mass Condition

Assume the critical source is in the cache. Let its readout mass be:

```text
p_* = p_{T,i*}
```

If:

```text
p_* >= epsilon_time
```

then the critical value contributes at least:

```text
||lambda_s p_* R v_*|| >= |lambda_s| epsilon_time sigma_min(R) ||v_*||
```

before cancellation by non-critical snapshots.

This lower bound has no factor of `gamma^(T-i*)`. The price is the explicit
snapshot budget `B_time`.

## 4. Logit-Margin Sufficient Condition

For exact token prediction, let the correct token be `y*`. Let the base model's
logit margin before snapshot readout be:

```text
Delta_base = logits_base[y*] - max_{y != y*} logits_base[y]
```

Let the snapshot branch add margin:

```text
Delta_snap =
lambda_s (W z_T)[y*] - max_{y != y*} lambda_s (W z_T)[y]
```

Then exact prediction is guaranteed if:

```text
Delta_base + Delta_snap > 0
```

This is the honest theorem behind the current synthetic success. A direct
snapshot-to-logit branch can solve exact-copy tasks when it creates enough
positive margin. It should not be confused with general reasoning, static
knowledge lookup, or fuzzy semantic retrieval.

## 5. Main Conditional Theorem

Under the following assumptions:

1. The critical token/source is captured: `i* in C`.
2. The snapshot budget is bounded: `|C| <= B_time`.
3. The readout assigns sufficient mass: `p_* >= epsilon_time`.
4. Projection and normalization are bounded and non-degenerate on the critical
   value direction.
5. For logits, the combined margin satisfies `Delta_base + Delta_snap > 0`.

Then the model has an exact-recall value path whose strength does not decay with
sequence distance `T - i*`.

The cost is:

```text
O(B_time d)
```

per readout, plus storage:

```text
O(B_time d)
```

This is not an `O(1)` pure RetNet context claim unless `B_time` is treated as a
fixed engineering constant.

## 6. Relation To Other Modules

```text
RetNet gate:
  slows or bounds decay of recurrent state components.

Milestone Snapshot Readout:
  preserves a queryable value path for exact token-time facts.

Block AttnRes:
  prevents depth-wise feature dilution over layer/block sources.

Engram:
  retrieves static lexical/factual memory by deterministic hash.
```

The snapshot readout is therefore an explicit time-axis module, not a depth-axis
Attention Residual and not an Engram table.

## 7. Experimental Obligations

Any positive result using this module must report:

```text
B_time
r_capture
p_*
snapshot_scale or logit_scale
module-drop delta
exact-match on held-out batches
non-copy reasoning ablations
```

Without these measurements, a copy-task win may be a shortcut rather than a
general reasoning improvement.

## 8. Current Evidence Boundary

The 2026-05-04 synthetic ablation supports only the exact-copy value-path claim:

```text
needle:      snapshot-to-logit improves held-out exact match
xor:         snapshot-to-logit improves loss but not clearly token accuracy
xor_final:   snapshot and snapshot-to-logit reach held-out exact match 1.0
alien:       no robust snapshot advantage
alien_static: too easy; RetNet also reaches held-out exact match 1.0
```

Therefore this theorem must remain scoped to exact token-time recall. Broader
reasoning and knowledge claims require separate task-specific theorems and
ablations.
