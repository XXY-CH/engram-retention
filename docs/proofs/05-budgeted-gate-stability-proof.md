# Formal Proof B1: Budgeted Gate Stability Under A Margin Gap

Created: 2026-05-03

Status: first proof draft for the sparse-gate objective.

Correction status: optional B-time proof. The active Dense RetNet + Block
AttnRes + Engram baseline does not require a sparse KV-anchor gate. Use this
only for future token-time anchor/session-cache extensions.

## 0. Claim Boundary

This proof studies an oracle/additive approximation to the gate objective. It is
not yet a proof that policy-gradient RL will learn the gate. It proves the shape
of the optimum once each candidate anchor has a well-defined utility.

## 1. Objects And Notation

For token `i`, let:

```text
a_i in {0,1}
```

where `a_i=1` means the token is written to the sparse residual cache.

Define the oracle utility:

```text
Delta_i = L_without_i - L_with_i
```

So `Delta_i > 0` means storing token `i` reduces task loss.

Under an additive approximation:

```text
L_task(a) = L_0 - sum_i a_i Delta_i
```

The penalized objective is:

```text
J_lambda(a) = L_task(a) + lambda sum_i a_i
            = L_0 + sum_i a_i(lambda - Delta_i)
```

## 2. Lemma B1: Threshold Form Of The Oracle Gate

**Statement**

For fixed utilities `Delta_i`, the minimizer of `J_lambda` is:

```text
a_i*(lambda) = 1[Delta_i > lambda]
```

with either value optimal when `Delta_i = lambda`.

**Proof**

The objective separates over tokens:

```text
J_lambda(a) = L_0 + sum_i a_i(lambda - Delta_i)
```

For each `i`:

- if `lambda - Delta_i < 0`, choosing `a_i=1` reduces the objective;
- if `lambda - Delta_i > 0`, choosing `a_i=0` reduces the objective;
- if `lambda = Delta_i`, both choices tie.

Therefore:

```text
a_i*(lambda) = 1[Delta_i > lambda]
```

QED.

## 3. Lemma B2: Stability Interval Requires A Margin Gap

Partition tokens into useful anchors `U` and non-useful anchors `Z`.

Assume a strict utility gap:

```text
Delta_min_useful = min_{i in U} Delta_i
Delta_max_useless = max_{i in Z} Delta_i
Delta_min_useful > Delta_max_useless
```

Then every:

```text
lambda in (Delta_max_useless, Delta_min_useful)
```

recovers exactly:

```text
a_i = 1 for i in U
a_i = 0 for i in Z
```

**Proof**

If `i in U`, then:

```text
Delta_i >= Delta_min_useful > lambda
```

so Lemma B1 selects `a_i=1`.

If `i in Z`, then:

```text
Delta_i <= Delta_max_useless < lambda
```

so Lemma B1 selects `a_i=0`.

QED.

## 4. Theorem B1: No Unconditional Stable Lambda

If the utility gap is absent, there is no nonempty open interval of `lambda`
that provably preserves the useful/useless partition for all tokens.

In particular, if:

```text
Delta_min_useful <= Delta_max_useless
```

then no threshold can separate all useful tokens from all useless tokens under
the oracle rule.

## 5. Budget-Constrained Lagrangian Form

Instead of fixing `lambda`, impose a cache budget:

```text
min E[L_task(a)]
s.t. E[sum_i a_i] <= B
```

The Lagrangian is:

```text
J_mu = E[L_task(a)] + mu(E[sum_i a_i] - B)
```

A primal-dual update:

```text
mu <- max(0, mu + eta(m - B))
```

acts as an adaptive memory-price controller:

- if cache usage `m` exceeds `B`, raise `mu`;
- if cache usage is below `B`, lower or keep `mu`.

This does not remove the margin-gap requirement for stable token identities, but
it avoids hand-tuning a fixed penalty for every task distribution.

## 6. Connection Back To A And C

A defines why an anchor matters:

```text
anchor useful if it creates a non-decaying gradient/information path
```

C defines when an external memory read is reliable:

```text
Engram useful if collision noise remains below the task margin
```

B turns both into a sparse selection problem:

```text
select anchors whose expected utility Delta_i exceeds the current memory price
```

Therefore the full proof program now has the following conditional chain:

```text
A: selected + read critical anchors bypass RetNet decay
C: hashed Engram reads are bounded perturbations under capacity conditions
B: a stable sparse gate exists only when utility margins separate useful from
   useless candidates
```
