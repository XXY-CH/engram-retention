# Formal Proof A3: Softmax Logit Margin Implies Readout Mass

Created: 2026-05-03

Status: sufficient-condition proof for the A1 assumption `p_{T,i} >= epsilon`.

Correction status: historical A-time softmax-mass proof for optional token-time
anchors. The same mathematical pattern is translated to depth sources in
`docs/proofs/17-depth-attnres-non-dilution-proof.md`.

## 0. Why A3 Is Needed

A1 assumes:

```text
p_{T,i} >= epsilon
```

for the critical cached anchor `i`. That assumption is not automatic. A3 gives a
plain sufficient condition in terms of the anchor's attention logit margin.

## 1. Setup

At read position `T`, let the sparse cache contain `m` anchors. Attention mass is:

```text
p_{T,j} = exp(s_j) / sum_{r in C_T} exp(s_r)
```

where:

```text
s_j = score_{T,j}
```

Let `i` be the critical anchor.

## 2. Theorem A3.1: Additive Logit Margin Lower-Bounds Attention Mass

**Statement**

If the critical anchor has margin `Delta > 0` over every other cached anchor:

```text
s_i >= s_j + Delta  for all j != i
```

then:

```text
p_{T,i} >= 1 / (1 + (m-1) exp(-Delta))
```

**Proof**

Write:

```text
p_{T,i}
= exp(s_i) / (exp(s_i) + sum_{j != i} exp(s_j))
= 1 / (1 + sum_{j != i} exp(s_j - s_i))
```

By the margin assumption:

```text
s_j - s_i <= -Delta
```

so:

```text
sum_{j != i} exp(s_j - s_i)
<= (m-1) exp(-Delta)
```

Therefore:

```text
p_{T,i}
>= 1 / (1 + (m-1) exp(-Delta))
```

QED.

## 3. Corollary A3.2: Required Margin For A Target Epsilon

To guarantee:

```text
p_{T,i} >= epsilon
```

it is sufficient that:

```text
1 / (1 + (m-1) exp(-Delta)) >= epsilon
```

Solving:

```text
Delta >= log((m-1) epsilon / (1 - epsilon))
```

for `0 < epsilon < 1`.

This bound is meaningful when the right-hand side is positive. If it is
negative, the target `epsilon` is below the uniform-cache baseline scale.

## 4. Consequence For Cache Growth

If the target readout mass `epsilon` is fixed while cache size `m` grows, the
required logit margin grows like:

```text
Delta = Omega(log m)
```

Thus A1's `p_{T,i} >= epsilon` condition is not free. For a large sparse cache,
the model must either:

```text
1. maintain a logit advantage that grows with log m;
2. reduce effective competition with routing / clustering / top-k prefiltering;
3. accept epsilon shrinking roughly as 1/m.
```

## 5. Relation To A2

A2 requires the value path to dominate cancellation terms:

```text
rho_min * epsilon * c_align > C_K + C_Q + C_R gamma^(T-i)
```

A3 says that `epsilon` can be guaranteed by a logit margin. Combining:

```text
rho_min * c_align
------------------ > C_K + C_Q + C_R gamma^(T-i)
1 + (m-1) exp(-Delta)
```

This is a precise tradeoff among:

```text
cache size m
readout margin Delta
residual scale rho_min
alignment c_align
cancellation bounds C_K, C_Q
RetNet decay gamma^(T-i)
```

No part of this proves the model learns the margin. It only states the condition
under which the readout-mass assumption is valid.
