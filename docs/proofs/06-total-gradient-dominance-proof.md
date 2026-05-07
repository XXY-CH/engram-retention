# Formal Proof A2: Total Gradient Requires Dominance, Not Just A Path

Created: 2026-05-03

Status: rigor pass after A1. This document prevents a common overclaim:
showing a non-decaying path exists is not the same as proving the total gradient
is non-decaying.

Correction status: historical A-time dominance proof for optional Sparse Anchor
Residuals. The depth-axis analogue for Kimi AttnRes is handled separately in
proof 17.

## 0. Why A2 Is Needed

A1 proved a pathwise statement:

```text
selected anchor + nonzero readout mass -> value path has no RetNet gamma^(T-i)
```

But the total gradient is a sum of multiple terms:

```text
g_i_total =
g_i_value
+ g_i_retnet
+ g_i_key
+ g_i_gate
+ g_i_other
```

A lower bound on `g_i_value` alone does not lower-bound `g_i_total`, because
other terms may point in the opposite direction. A2 states the extra dominance
condition needed to upgrade A1 to a total-gradient claim.

## 1. Notation

Fix a unit direction `u_i`. Define scalar directional components:

```text
v = <g_i_value, u_i>
r = <g_i_retnet, u_i>
k = <g_i_key, u_i>
q = <g_i_gate + g_i_other, u_i>
```

Then:

```text
<g_i_total, u_i> = v + r + k + q
```

From A1:

```text
|v| >= A
A = rho_min * epsilon * c_align
```

From RetNet decay:

```text
|r| <= C_R gamma^(T-i)
```

Assume the non-value residual terms have known directional bounds:

```text
|k| <= C_K
|q| <= C_Q
```

These are not automatic. They require either analysis of the attention logits,
straight-through gate estimator, stop-gradient gate design, or empirical
Jacobian bounds.

## 2. Lemma A2.1: Path Existence Does Not Imply Total Non-Decay

**Statement**

Even if `|v| >= A > 0`, the total directional gradient can be zero.

**Proof**

Choose:

```text
r = 0
k = -v
q = 0
```

Then:

```text
<g_i_total, u_i> = v + r + k + q = 0
```

This satisfies the existence of a nonzero value path but cancels in the total
directional gradient.

QED.

## 3. Theorem A2: Dominance Condition For Total Directional Non-Decay

**Statement**

If:

```text
A > C_K + C_Q + C_R gamma^(T-i)
```

then:

```text
|<g_i_total, u_i>|
>= A - C_K - C_Q - C_R gamma^(T-i)
> 0
```

Therefore the total directional gradient is nonzero and remains bounded away
from zero when the value path dominates all non-value directional terms.

**Proof**

By the reverse triangle inequality:

```text
|v + r + k + q|
>= |v| - |r| - |k| - |q|
```

Use the bounds:

```text
|v| >= A
|r| <= C_R gamma^(T-i)
|k| <= C_K
|q| <= C_Q
```

Then:

```text
|<g_i_total, u_i>|
>= A - C_R gamma^(T-i) - C_K - C_Q
```

If the right side is positive, the total directional gradient cannot vanish.

QED.

## 4. Consequence

The rigorous A-side claim is now:

```text
A1: a selected/read sparse residual anchor creates a non-decaying value path.
A2: the total gradient is non-decaying only if that value path dominates
    key-path, gate-path, and other cancellation terms.
```

This suggests two architecture-safe choices:

1. Stop-gradient through the anchor-selection gate during value-path analysis,
   so `g_i_gate` is not mixed into the same theorem.
2. Instrument key-path and total-gradient directions separately in experiments.

The minimal empirical logging should include:

```text
value-path directional gradient
key-path directional gradient
gate-path directional gradient
total directional gradient
cosine(value-path, total-gradient)
```

Without this separation, an experiment may observe healthy value paths while
the total training signal still cancels.
