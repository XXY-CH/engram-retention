# Formal Proof A1: Conditional Non-Decaying Gradient Path

Created: 2026-05-03

Status: first formal proof draft. This proves a pathwise statement and records
the extra assumptions needed before upgrading it to a total-gradient theorem.

Correction status: historical A-time proof for the optional Sparse Anchor
Residual. It remains mathematically useful if token-time anchors are
reintroduced, but it is not the Kimi Block AttnRes theorem.

## 0. Claim Boundary

This proof does not show that optional Sparse Anchor Residuals automatically solve
long-context reasoning. It shows a narrower but important fact:

```text
RetNet-only contribution from token i to token T decays like gamma^(T-i).
If token i is stored as a sparse residual anchor and later receives attention
mass at least epsilon, the residual value path from L_T to h_i does not contain
gamma^(T-i).
```

The actual architecture must still learn:

```text
gate recall:        a_i = 1 for critical tokens
future readout:     p_{T,i} >= epsilon
budget discipline:  |C_T| <= B
```

## 1. Objects And Notation

Let `h_t in R^d` be the hidden state at token `t`.

The RetNet recurrent state is:

```text
S_t = Gamma S_{t-1} + k_t v_t^T
```

Assume the recurrence is contractive in an operator norm:

```text
||Gamma||_op <= gamma < 1
```

This is stronger than merely requiring `rho(Gamma) < 1`, but it is the clean
assumption needed for a one-line finite-time bound. If only spectral radius is
known, the proof must add a non-normality constant or use `gamma + eta`.

PDF audit note: this matches the RetNet paper's fixed per-head decay values
below one. The implementation must preserve this contract; an unconstrained
trainable parameter such as `abs(raw_gamma)` is insufficient because it can
exceed one after training.

The retention readout is:

```text
r_t = q_t S_t
```

The sparse residual cache stores selected anchors:

```text
a_i in {0, 1}
key_i = K_a h_i
val_i = V_a h_i
C_T = {(key_j, val_j) : j <= T, a_j = 1}
```

At position `T`, residual attention reads:

```text
z_T = sum_{j: a_j=1} p_{T,j} V_a h_j
```

where:

```text
p_{T,j} = softmax_j(score_{T,j})
score_{T,j} = (Q_a h_T)^T K_a h_j / sqrt(d_a)
```

The local output before loss is:

```text
u_T = h_T + F(r_T) + rho_T z_T
y_T = N(u_T)
L_T = ell(y_T)
```

Here `N` is a normalization map, and `rho_T` is the scalar residual-cache gate.

## 2. Assumptions

**A-RetNet bounded local maps**

There exists `C_local` such that the Jacobians of `q_t`, `k_t`, `v_t`, `F`, and
the downstream maps on the RetNet path are bounded by `C_local` over the region
being analyzed.

**A-Residual selection**

The critical token `i` is written:

```text
a_i = 1
```

**A-Residual readout mass**

At the later token `T`, the critical anchor receives non-negligible mass:

```text
p_{T,i} >= epsilon > 0
```

**A-Residual scale**

The residual-cache scale is bounded away from zero:

```text
rho_T >= rho_min > 0
```

**A-Value path non-degeneracy**

Let:

```text
g_T = dL_T/dy_T
J_N = dy_T/du_T
```

There is a useful direction `u_i` with `||u_i|| = 1` and a constant
`c_align > 0` such that:

```text
|<V_a^T J_N^T g_T, u_i>| >= c_align
```

This is the honest alignment assumption. Without it, a nonzero path exists but
the directional lower bound can be zero because the loss gradient, normalization
Jacobian, projection, and chosen direction may be orthogonal.

## 3. Lemma 1: RetNet Path Decays With Distance

**Statement**

Under `||Gamma||_op <= gamma < 1` and bounded local maps, the contribution from
the token-`i` retention update to the token-`T` loss through the RetNet recurrent
state is bounded by:

```text
||(dL_T/dh_i)_RetNet|| <= C_R gamma^(T-i)
```

for some constant `C_R` independent of `T-i`.

**Proof**

Unroll the recurrent state:

```text
S_T = Gamma^(T-i) S_i + sum_{s=i+1}^{T} Gamma^(T-s) k_s v_s^T
```

The portion of `S_T` caused by the update at `i` contains the factor:

```text
Gamma^(T-i) k_i v_i^T
```

Taking operator norms:

```text
||Gamma^(T-i)||_op <= ||Gamma||_op^(T-i) <= gamma^(T-i)
```

All maps around this factor are bounded by assumption, so their product can be
absorbed into a constant `C_R`. Therefore:

```text
||(dL_T/dh_i)_RetNet|| <= C_R gamma^(T-i)
```

As `T-i -> infinity`, this contribution goes to zero.

QED.

## 4. Lemma 2: Residual Value Path Has No RetNet Decay Factor

**Statement**

Condition on `a_i = 1`. The value part of the sparse residual gradient from
`L_T` to `h_i` is:

```text
(dL_T/dh_i)_value = rho_T p_{T,i} V_a^T J_N^T g_T
```

where the attention weights are treated as fixed for this pathwise derivative.
This expression has no `Gamma^(T-i)` factor.

**Proof**

The sparse residual readout is:

```text
z_T = sum_{j: a_j=1} p_{T,j} V_a h_j
```

For the value path, hold the softmax weights fixed and differentiate only
through the stored value `V_a h_i`:

```text
partial z_T / partial h_i = p_{T,i} V_a
```

Since:

```text
u_T = h_T + F(r_T) + rho_T z_T
y_T = N(u_T)
L_T = ell(y_T)
```

the chain rule gives:

```text
(dL_T/dh_i)_value
= (partial z_T / partial h_i)^T rho_T J_N^T g_T
= rho_T p_{T,i} V_a^T J_N^T g_T
```

No recurrent transition from `i` to `T` appears in this path; the anchor is read
directly from `C_T`. Therefore this path does not contain `Gamma^(T-i)`.

QED.

## 5. Theorem A1: Conditional Non-Decaying Directional Gradient

**Statement**

Under the assumptions above, the value-path directional gradient satisfies:

```text
|< (dL_T/dh_i)_value, u_i >|
>= rho_min * epsilon * c_align
```

and the combined RetNet plus value-path directional signal obeys:

```text
|< (dL_T/dh_i)_value, u_i >|
- |< (dL_T/dh_i)_RetNet, u_i >|
>= rho_min * epsilon * c_align - C_R gamma^(T-i)
```

Thus for sufficiently large `T-i`, the RetNet component vanishes while the
residual value path remains bounded away from zero.

**Proof**

From Lemma 2:

```text
(dL_T/dh_i)_value = rho_T p_{T,i} V_a^T J_N^T g_T
```

Take inner product with `u_i`:

```text
|< (dL_T/dh_i)_value, u_i >|
= rho_T p_{T,i} |<V_a^T J_N^T g_T, u_i>|
```

Using:

```text
rho_T >= rho_min
p_{T,i} >= epsilon
|<V_a^T J_N^T g_T, u_i>| >= c_align
```

we get:

```text
|< (dL_T/dh_i)_value, u_i >|
>= rho_min * epsilon * c_align
```

From Lemma 1 and `||u_i|| = 1`:

```text
|< (dL_T/dh_i)_RetNet, u_i >|
<= ||(dL_T/dh_i)_RetNet||
<= C_R gamma^(T-i)
```

Combining the two bounds gives:

```text
|< (dL_T/dh_i)_value, u_i >|
- |< (dL_T/dh_i)_RetNet, u_i >|
>= rho_min * epsilon * c_align - C_R gamma^(T-i)
```

Because `0 < gamma < 1`, the second term goes to zero as `T-i -> infinity`.

QED.

## 6. Cache-Size Dependence

The theorem hides the hardest practical condition in:

```text
p_{T,i} >= epsilon
```

If the cache has `m` anchors and attention is unstructured, a natural baseline is:

```text
p_{T,i} ~= 1/m
```

If logits are uniformly bounded:

```text
score_{T,j} in [-L, L]
```

then every selected anchor has a crude lower bound:

```text
p_{T,i} >= exp(-L) / (m exp(L)) = exp(-2L) / m
```

This requires bounded query/key norms or explicit logit clipping/temperature
control. Raw dot-product attention does not guarantee the bound by itself.

This means the gradient lower bound can degrade with cache size:

```text
rho_min * c_align * exp(-2L) / m
```

So sparse residual attention does not remove the memory budget problem. It moves
the problem from distance `T-i` to cache quality and cache size `m`.

The useful research target is therefore:

```text
m << T
and
p_{T,i} remains non-negligible for critical anchors
```

not:

```text
m = O(1) for all tasks
```

## 7. Stochastic Gate Corollary

Let `critical(i)` mean token `i` is necessary for the prediction at `T`.

Assume:

```text
Pr(a_i = 1 | critical(i)) >= r
Pr(p_{T,i} >= epsilon | a_i = 1, critical(i)) >= s
```

Then the non-decaying value path exists with probability at least:

```text
r * s
```

Under the same alignment assumption, the event-level directional lower bound is:

```text
rho_min * epsilon * c_align
```

The expected lower bound needs an additional sign/alignment condition across
examples; otherwise positive and negative directional contributions can cancel.

## 8. Consequence For The Research Program

The first mathematical bottleneck is now explicit:

```text
The architecture's long-context guarantee is conditional on gate recall and
attention readout mass, not on RetNet recurrence alone.
```

This suggests the next two proof obligations:

1. Define a measurable anchor utility `Delta_i` that predicts whether `a_i`
   should be one.
2. Prove or empirically validate that the gate can maintain high recall under
   budget `B`.

The proof also defines the first experiment:

```text
Measure ||dL_T/dh_i|| over increasing T-i for:
1. RetNet-only path.
2. RetNet + forced anchor at i.
3. RetNet + learned/stochastic gate.
```

Expected signature:

```text
RetNet-only:          exponential decay in T-i
forced anchor:        distance-stable value-path signal if p_{T,i} is stable
learned gate:         depends on critical-token recall and p_{T,i}
```
