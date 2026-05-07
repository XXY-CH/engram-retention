# Milestone-Triggered Retention Gate

Created: 2026-05-04

Status: conditional A-Time theorem for the optional milestone-gated RetNet
extension. This proof is not part of the original RetNet paper and is not Kimi
Attention Residuals.

## 0. Goal

The active Dense baseline previously excluded token-time preservation gates. If
the architecture adopts explicit thought milestones such as:

```text
<MARK_THOUGHT>
<CRITICAL_ASSUMPTION>
```

then the time-axis proof target becomes:

```text
critical reasoning state should survive many future steps without requiring a
Transformer-style KV cache.
```

This proof gives a sufficient condition. It does not prove that RL will learn
when to emit milestones.

## 1. Minimal Gated Retention Model

Let a protected state component evolve as:

```text
s_t = g_t s_{t-1} + u_t
```

where:

```text
s_t  = protected RetNet state component
u_t  = new update
g_t  in [0,1] = data-dependent retention gate
```

For standard fixed RetNet, `g_t = gamma < 1`. For a milestone-protected interval,
the design intends:

```text
g_t >= 1 - epsilon_t
```

for steps where the milestone remains relevant.

## 2. Lemma: Cumulative Decay, Not Per-Step Slogans

For a milestone written at time `i`, its contribution to time `T` is multiplied
by:

```text
G_{i:T} = product_{t=i+1}^{T} g_t
```

If:

```text
g_t >= 1 - epsilon_t
0 <= epsilon_t <= 1/2
sum_{t=i+1}^{T} epsilon_t <= eta
```

then:

```text
G_{i:T} >= exp(-2 eta)
```

Proof sketch:

```text
log G_{i:T}
= sum_t log(g_t)
>= sum_t log(1 - epsilon_t)
>= -2 sum_t epsilon_t
>= -2 eta
```

So the milestone path is non-vanishing only when cumulative leakage is bounded.
This is stronger and more precise than saying `gamma = 0.9999`.

## 3. Theorem: Conditional Time-Axis Preservation

Assume:

1. A critical milestone is emitted at time `i`.
2. The milestone controls a protected state subspace `P`.
3. On `P`, the retention gate satisfies:

```text
sum_{t=i+1}^{T} (1 - g_t) <= eta
```

4. Additive updates orthogonal or competing with the protected component are
bounded by `C_update`.
5. Normalization and Dense blocks do not erase the protected direction.

Then the protected milestone contribution at time `T` has lower bound:

```text
||P s_T|| >= exp(-2 eta) ||P s_i|| - C_update
```

Thus the time-axis protected component does not decay exponentially with
`T-i`; it decays with cumulative allowed leakage `eta`.

## 4. Why Absolute Permanent Pass-Through Is Dangerous

Setting:

```text
g_t = 1
```

forever avoids decay but creates state saturation:

```text
s_T = s_i + sum_{t=i+1}^{T} u_t
```

Unless updates are controlled, the protected state can become a sum of unrelated
old milestones. Therefore the implementation needs one or more of:

```text
finite milestone budget
protected subspace allocation
age or priority decay at readout
overwrite policy
false-positive penalty for milestone spam
```

## 5. Milestone Trigger Quality

Let:

```text
r_m = Pr(critical milestone is emitted)
f_m = Pr(non-critical milestone is emitted)
```

The gate is useful only if:

```text
r_m is high enough for recall
f_m is low enough to avoid state saturation
```

A minimal training/evaluation objective must measure both:

```text
milestone recall on planted critical assumptions
milestone precision / spam rate on distractors
```

This replaces the old sparse-cache budget proof for the milestone-gated route.

## 6. Relation To Block AttnRes

Block AttnRes does not by itself preserve token-time facts. It can only read the
layer/block summaries available to it.

The intended division is:

```text
Milestone gate:
  slows time-axis decay of critical state components.

Block AttnRes:
  gives deep layers direct access to earlier depth summaries of those components.
```

So the system has two separate conditions:

```text
time condition:  cumulative leakage eta is bounded
depth condition: beta_{source->layer} >= epsilon_depth
```

Both are required for the strong long-CoT story.

## 7. Open Implementation Contract

The first implementation should not use unconstrained permanent pass-through.
It should include:

```text
explicit milestone token ids or hidden monitor head
negative/default-closed gate bias
bounded milestone budget per window
minimum distance or anti-spam regularizer
logging of r_m, f_m, cumulative leakage eta
stress tests where old assumptions remain needed after many distractors
```

This theorem is conditional on the trigger and gate satisfying those measured
conditions.
