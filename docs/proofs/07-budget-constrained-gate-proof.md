# Formal Proof B2: Hard-Budget Gate Selects Top Utilities

Created: 2026-05-03

Status: budget-constrained companion to B1.

Correction status: optional B-time proof for future sparse-anchor/cache gates.
It is not part of the active Dense baseline unless token-time anchors are
reintroduced.

## 0. Claim Boundary

B1 studied the penalized objective:

```text
J_lambda(a) = L_task(a) + lambda sum_i a_i
```

B2 studies a hard cache budget:

```text
sum_i a_i <= B
```

The result is still oracle-level: it assumes each token has a fixed utility
`Delta_i`. It does not prove an RL algorithm learns those utilities.

## 1. Objective

Using the same additive approximation as B1:

```text
L_task(a) = L_0 - sum_i a_i Delta_i
```

the hard-budget problem is:

```text
min_a L_0 - sum_i a_i Delta_i
s.t. a_i in {0,1}
     sum_i a_i <= B
```

Equivalently:

```text
max_a sum_i a_i Delta_i
s.t. a_i in {0,1}
     sum_i a_i <= B
```

## 2. Theorem B2: Top-Positive-Utilities Are Optimal

**Statement**

Sort utilities in non-increasing order:

```text
Delta_(1) >= Delta_(2) >= ... >= Delta_(n)
```

An optimal solution selects exactly:

```text
S* = {i among the top B utilities with Delta_i > 0}
```

Ties at the boundary can produce multiple optimal solutions.

**Proof**

First, no optimal solution selects a token with `Delta_i < 0`, because setting
its `a_i` from `1` to `0` increases the maximization objective by `-Delta_i > 0`
and also relaxes the budget.

Second, suppose an optimal solution selects `j` but does not select `i`, with:

```text
Delta_i > Delta_j
```

and `Delta_i > 0`. Swap the selection:

```text
a_i <- 1
a_j <- 0
```

The budget is unchanged, and the objective increases by:

```text
Delta_i - Delta_j > 0
```

contradicting optimality. Therefore every selected lower-utility item can be
replaced by an unselected higher-utility positive item. The optimum must select
the highest positive utilities up to budget `B`.

QED.

## 3. Relation To Lambda Thresholding

If there is a strict gap at the budget boundary:

```text
Delta_(B) > Delta_(B+1)
```

and:

```text
Delta_(B) > 0
```

then any:

```text
lambda in (Delta_(B+1), Delta_(B))
```

makes the penalized B1 threshold select the same top-`B` set:

```text
a_i*(lambda) = 1[Delta_i > lambda]
```

If no such gap exists, a fixed `lambda` can be unstable at the boundary.

## 4. Consequence For Training

The stable gate problem is not just "choose lambda." It is:

```text
estimate Delta_i well enough
and maintain a margin at the budget boundary
```

For the proposed architecture, `Delta_i` should be measured or approximated by
the value of preserving token `i` as a future anchor:

```text
Delta_i = L_without_anchor_i - L_with_anchor_i
```

This connects back to A2: if the value path exists but cancels in the total
gradient, the estimated `Delta_i` may be small even for a token that looked
important pathwise. Therefore B must consume utility estimates tied to actual
loss improvement, not merely attention salience.
