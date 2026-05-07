# Reasoning State Reuse Theorem

Created: 2026-05-04

Status: future theorem track for expanding beyond exact token recall.

## Claim Boundary

Milestone snapshots currently prove an exact token-time value path. To claim
general reasoning enhancement, the captured object must be an intermediate
reasoning state rather than a raw answer token.

## Target Statement

Let `r_i` be an intermediate reasoning state at milestone `i`, such as:

```text
premise summary
variable binding
intermediate conclusion
branch choice
contradiction marker
```

If:

1. `r_i` is captured with probability at least `r_capture`.
2. Later query state `q_T` assigns readout mass at least `epsilon_reason`.
3. The downstream Dense/RetNet computation is Lipschitz and preserves a margin
   for the correct final decision when conditioned on `r_i`.
4. Distractor states do not cancel the useful direction beyond margin `m`.

Then the final reasoning margin has a non-decaying contribution from `r_i`:

```text
Margin_T >= F_margin(r_i, q_T) - cancellation
```

with no `gamma^(T-i)` factor.

## Required Evidence

This theorem is not supported by needle-copy results. It requires tasks where:

```text
answer != copied milestone token
answer depends on composing one or more captured intermediate states
module-drop(snapshot) reduces reasoning accuracy
```

Examples:

```text
xor_final
tree logic
variable binding
multi-step arithmetic with hidden intermediate summaries
counterfactual premise selection
```

## Non-Claim

Snapshot-to-logit success on exact copy does not imply this theorem.
