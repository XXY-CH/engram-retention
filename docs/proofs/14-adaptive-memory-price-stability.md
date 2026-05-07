# Formal Proof B4: Adaptive Memory-Price Stability

Created: 2026-05-03

Status: turns the informal Lagrangian memory-price controller from B1 into a
discrete stability statement under a strict budget-boundary gap.

Correction status: optional B-time proof for future sparse-anchor/session-cache
budget controllers. It is not part of the active Dense baseline.

## 0. Claim Boundary

B1 proposed the update:

```text
mu_{t+1} = max(0, mu_t + eta (m(mu_t) - B))
```

where `mu` is the memory price and `m(mu)` is cache usage under threshold
selection. This note does not prove policy-gradient RL convergence. It proves
that the oracle memory-price controller reaches a stable budget interval when
utilities are fixed and have a strict boundary gap.

## 1. Setup

Let fixed oracle utilities be sorted:

```text
Delta_(1) >= Delta_(2) >= ... >= Delta_(n)
```

Assume the budget boundary is strict and positive:

```text
Delta_(B) > Delta_(B+1)
Delta_(B) > 0
```

Define:

```text
l = Delta_(B+1)
u = Delta_(B)
w = u - l > 0
```

Under threshold selection:

```text
a_i(mu) = 1[Delta_i > mu]
m(mu) = sum_i a_i(mu)
```

Every:

```text
mu in (l, u)
```

selects exactly `B` anchors, so:

```text
m(mu) = B
```

## 2. Theorem B4: Finite-Time Entry Into Stable Budget Interval

**Statement**

Assume:

```text
0 < eta n < w
```

and use:

```text
mu_{t+1} = max(0, mu_t + eta(m(mu_t)-B))
```

Then from any initial `mu_0 >= 0`, the iterates enter the interval:

```text
(l, u)
```

in finite time, and once inside, remain fixed.

**Proof**

First, if:

```text
mu_t in (l, u)
```

then `m(mu_t)=B`, so:

```text
mu_{t+1}=mu_t
```

Thus the interval is pointwise fixed.

Now consider `mu_t <= l`. Since `mu_t` is below the budget threshold, at least
the top `B+1` utilities exceed `mu_t` except at equality/tie edge cases. With a
strict tie convention or an arbitrarily small perturbation away from exact
ties:

```text
m(mu_t) > B
```

so:

```text
mu_{t+1} > mu_t
```

The update is bounded by:

```text
mu_{t+1} - mu_t = eta(m(mu_t)-B) <= eta n
```

When the increasing sequence first crosses `l`, the previous iterate satisfies
`mu_t <= l`, so:

```text
mu_{t+1} <= l + eta n < l + w = u
```

and:

```text
mu_{t+1} > l
```

Thus it enters `(l,u)`.

Now consider `mu_t >= u`. Then fewer than `B` utilities exceed `mu_t`, so:

```text
m(mu_t) < B
```

and:

```text
mu_{t+1} < mu_t
```

The downward step size is bounded by:

```text
mu_t - mu_{t+1} = eta(B-m(mu_t)) <= eta n
```

When the decreasing sequence first crosses `u`, the previous iterate satisfies
`mu_t >= u`, so:

```text
mu_{t+1} >= u - eta n > u - w = l
```

and:

```text
mu_{t+1} < u
```

Thus it enters `(l,u)`.

In all cases, finite monotone movement toward the interval occurs with steps of
at least `eta` whenever `m(mu_t)-B` is a nonzero integer, until crossing into
the interval. Once inside, the update is zero.

QED.

## 3. Tie And Boundary Caveats

If:

```text
Delta_(B) = Delta_(B+1)
```

then `w=0`, and no open stable interval exists. This matches B1/B2/B3: stable
budget selection requires a margin at the boundary.

If `mu_t` lands exactly on a utility value, the rule:

```text
a_i(mu)=1[Delta_i>mu]
```

may create a one-step discontinuity. This can be handled by:

```text
deterministic tie-breaking
small random tie perturbation
or using a soft gate during training and hardening at inference
```

The theorem is about the strict-margin region, not the tie surface.

## 4. Stochastic Or Learned Utilities

If the gate uses estimated utilities `Delta_hat_i`, B3 applies first. Suppose:

```text
|Delta_hat_i - Delta_i| <= eta_Delta
```

and:

```text
Delta_(B) - Delta_(B+1) > 2 eta_Delta
```

Then the estimated top-budget set equals the oracle set. The stable interval for
estimated utilities has width at least:

```text
w_hat >= w - 2 eta_Delta
```

The memory-price step should satisfy:

```text
eta n < w - 2 eta_Delta
```

to avoid jumping across the estimated stable interval.

## 5. Consequence For RL Framing

The rigorous claim is not:

```text
RL will automatically learn the right sparse cache.
```

The rigorous claim is:

```text
if utility estimates have a strict boundary margin and the price step is small
relative to that margin, the adaptive memory price has a stable budget interval.
```

This makes the RL component a measurable control problem:

```text
measure utility boundary gap
measure utility-estimation error
set eta so eta n < gap - 2 error
```

If this inequality fails, oscillation or unstable cache membership is expected,
not surprising.
