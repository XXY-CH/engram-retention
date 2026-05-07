# Formal Proof C1: Engram Hash Collision And Noise Bounds

Created: 2026-05-03

Status: first proof draft for the Engram capacity question. This analyzes a
static multi-head hash table, not the current `EngramGate` implementation.

PDF audit note: the signed zero-mean estimator is primarily supported by the
Feature Hashing paper. Count-Min Sketch supports the multi-hash sketch pattern
and pairwise-independence theme, but it is not the same estimator because it uses
nonnegative counters and a min query rule.

## 0. Claim Boundary

The goal is to prove conditional capacity bounds for a proposed static Engram
memory:

```text
query key x
-> K independent hash heads
-> M slots per head
-> signed / zero-mean vector aggregation
-> gated residual injection into hidden state
```

This proof does not claim that hashing creates semantic memory. It only bounds
collision probability and collision-noise variance under explicit independence
and bounded-vector assumptions.

## 1. Objects And Notation

Let:

```text
N = number of active Engram keys stored
M = number of slots per hash head
K = number of independent hash heads
d = embedding dimension
```

Each stored key `x` has target vector:

```text
e_x in R^d
```

Assume:

```text
||e_x||_2 <= R
```

For each head `k`, use:

```text
h_k: keys -> {1, ..., M}
s_k: keys -> {-1, +1}
```

where `h_k` is pairwise independent across keys and independent across heads.
The signs `s_k` are independent Rademacher signs across keys and heads. This is
slightly stronger than the minimal condition needed for every term, but it keeps
the variance proof clean and matches the signed feature-hashing idealization.

The table for head `k` stores:

```text
T_k[b] = sum_{x: h_k(x)=b} s_k(x) e_x
```

The retrieval estimate for query `x` is:

```text
y_x = (1/K) sum_{k=1}^K s_k(x) T_k[h_k(x)]
```

Expanding:

```text
y_x = e_x + eta_x
```

where collision noise is:

```text
eta_x =
(1/K) sum_{k=1}^K sum_{x' != x}
1[h_k(x') = h_k(x)] s_k(x) s_k(x') e_{x'}
```

## 2. Lemma C1: Single-Head Collision Probability

**Statement**

For a fixed query key `x`, the probability that at least one of the other
`N-1` active keys collides with it in one head is bounded by:

```text
Pr(any collision in one head)
<= (N-1)/M
```

and is approximately:

```text
1 - (1 - 1/M)^(N-1)
```

under full independence.

**Proof**

For any other key `x'`:

```text
Pr(h_k(x') = h_k(x)) = 1/M
```

By the union bound:

```text
Pr(exists x' != x: h_k(x') = h_k(x))
<= sum_{x' != x} Pr(h_k(x') = h_k(x))
= (N-1)/M
```

Under full independence, no collision across all `N-1` other keys has
probability `(1 - 1/M)^(N-1)`, giving the approximation above.

QED.

## 3. Lemma C2: All-Head Consistent False Collision

**Statement**

For a fixed wrong key `x' != x`, the probability that it collides with `x` in
all `K` heads is:

```text
Pr(for all k, h_k(x') = h_k(x)) = 1/M^K
```

The probability that any wrong key collides with `x` in all heads is bounded by:

```text
Pr(any all-head collision) <= (N-1)/M^K
```

**Proof**

Independence across heads gives:

```text
prod_{k=1}^K Pr(h_k(x') = h_k(x)) = (1/M)^K
```

Union bound over `N-1` wrong keys:

```text
Pr(exists x' != x colliding in all heads)
<= (N-1)/M^K
```

QED.

This all-head bound controls exact consistent false matches only. It does not
control all retrieval noise, because one-head collisions still contribute to the
averaged vector. Lemma C4 handles aggregate partial-collision noise separately.

## 4. Lemma C3: Signed Collision Noise Is Zero-Mean

**Statement**

For the signed multi-head estimate:

```text
y_x = e_x + eta_x
```

the collision noise satisfies:

```text
E[eta_x] = 0
```

**Proof**

For every wrong key `x'`, each collision term is:

```text
1[h_k(x') = h_k(x)] s_k(x) s_k(x') e_{x'}
```

The indicator is independent of the sign product, and:

```text
E[s_k(x) s_k(x')] = 0
```

because `s_k(x)` and `s_k(x')` are independent Rademacher signs. Therefore each
wrong-key term has zero expectation, and linearity of expectation gives:

```text
E[eta_x] = 0
```

QED.

## 5. Lemma C4: Collision Noise Variance Bound

**Statement**

Under the hash/sign independence assumptions above and `||e_x||_2 <= R`, the
expected squared noise norm is bounded by:

```text
E[||eta_x||_2^2] <= ((N-1) R^2) / (K M)
```

**Proof Sketch**

For one head, define:

```text
eta_{x,k} = sum_{x' != x}
1[h_k(x') = h_k(x)] s_k(x) s_k(x') e_{x'}
```

The multi-head average is:

```text
eta_x = (1/K) sum_k eta_{x,k}
```

Each wrong-key term has zero mean by Lemma C3. Cross terms vanish in expectation
because of independent signs. Therefore:

```text
E[||eta_{x,k}||_2^2]
= sum_{x' != x} Pr(h_k(x') = h_k(x)) ||e_{x'}||_2^2
<= (N-1) R^2 / M
```

Averaging `K` independent heads reduces variance by `K`:

```text
E[||eta_x||_2^2]
= (1/K^2) sum_k E[||eta_{x,k}||_2^2]
<= ((N-1) R^2) / (K M)
```

QED.

## 6. Theorem C1: Two Capacity Conditions

To make static Engram retrieval usable, two different risks must be controlled.

**Consistent false collision**

For failure probability at most `delta`:

```text
(N-1)/M^K <= delta
```

so a sufficient condition is:

```text
M^K >= (N-1)/delta
```

**Noise energy**

For expected noise energy at most `tau^2`:

```text
((N-1) R^2)/(K M) <= tau^2
```

so a sufficient condition is:

```text
K M >= ((N-1) R^2)/tau^2
```

Therefore the Engram capacity condition is not just:

```text
M^K >> N
```

It also needs:

```text
K M >> N * R^2 / tau^2
```

The first controls rare exact false matches. The second controls aggregate
collision noise from many partial collisions.

## 7. Multimodal Namespace Condition

If text n-grams, visual VQ codes, audio codes, and sketch/entity tags share the
same hash namespace, their active key count pools together:

```text
N_total = N_text + N_image + N_audio + N_sketch + ...
```

The bounds above then use `N_total`, increasing both:

```text
(N_total - 1)/M^K
```

and:

```text
((N_total - 1) R^2)/(K M)
```

A clean design should salt or namespace the key:

```text
hash_key = Hash(modality_id, position_or_sketch, semantic_tag, ngram_or_vq_span)
```

This does not remove collisions inside a namespace, but it prevents unrelated
modalities from unnecessarily sharing the same collision pool.

## 8. Consequence For The Research Program

Engram is viable as a static memory only if the active key set is controlled.
The right scaling variable is:

```text
N = active stored keys
```

not the theoretical vocabulary size.

The next proof step is to connect this noise bound to hidden-state residual
injection:

```text
h <- h + alpha e_hat
e_hat = e + eta
```

If `alpha` is bounded and `E[||eta||^2]` is small enough relative to the hidden
state signal margin, then Engram noise can be treated as a bounded perturbation.
That perturbation theorem is the bridge from C back to model correctness.
