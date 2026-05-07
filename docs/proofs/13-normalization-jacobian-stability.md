# Formal Proof A5: Normalization Jacobian Stability

Created: 2026-05-03

Status: closes the normalization assumption used by A1/A2/G1.

Correction status: shared support proof. The sparse-anchor examples are
historical, but the normalization caveat also applies to depth-wise Block
AttnRes and Engram residual injection.

## 0. Claim Boundary

A1 assumed a non-degenerate normalized residual path:

```text
|<V_a^T J_N^T g_T, u_i>| >= c_align
```

where `J_N` is the Jacobian of the normalization map after residual fusion.
This note clarifies when such an assumption is mathematically legal.

The result is conditional:

```text
normalization does not destroy the residual gradient if the activation norm or
variance is bounded away from zero and the useful gradient direction is not in
the normalization nullspace.
```

LayerNorm/RMSNorm can have null or near-null directions. Therefore non-decay
cannot be claimed without an explicit alignment or subspace condition.

## 1. RMSNorm Jacobian

For `x in R^d`, define RMSNorm without affine scale:

```text
r(x) = sqrt((1/d)||x||_2^2 + eps)
N(x) = x / r(x)
```

The Jacobian is:

```text
J_RMS(x)
= (1/r) I - (1/(d r^3)) x x^T
= (1/r) [I - x x^T/(d r^2)]
```

## 2. Lemma A5.1: RMSNorm Operator Upper Bound

**Statement**

If:

```text
r(x) >= r_min > 0
```

then:

```text
||J_RMS(x)||_op <= 1/r_min
```

**Proof**

The matrix:

```text
P = I - x x^T/(d r^2)
```

has eigenvalue `1` on directions orthogonal to `x`. Along the direction of
`x`, its eigenvalue is:

```text
1 - ||x||_2^2/(d r^2)
= eps / r^2
```

which lies in `[0,1]` for `eps >= 0`. Thus:

```text
||P||_op <= 1
```

and:

```text
||J_RMS(x)||_op <= 1/r <= 1/r_min
```

QED.

## 3. Lemma A5.2: RMSNorm Lower Bound On Useful Subspaces

**Statement**

Let `w` be a vector decomposed into:

```text
w = w_parallel + w_perp
```

where `w_parallel` is parallel to `x` and `w_perp` is orthogonal to `x`.

Then:

```text
||J_RMS(x) w||_2^2
= (eps^2/r^6) ||w_parallel||_2^2
  + (1/r^2) ||w_perp||_2^2
```

Therefore, if:

```text
r(x) <= r_max
||w_perp||_2 >= beta ||w||_2
```

then:

```text
||J_RMS(x) w||_2 >= (beta/r_max) ||w||_2
```

**Proof**

From the eigen decomposition above, `J_RMS` has eigenvalue:

```text
eps/r^3
```

on the radial direction and eigenvalue:

```text
1/r
```

on the orthogonal subspace. Squaring and summing orthogonal components gives
the equality. Dropping the nonnegative radial term gives:

```text
||J_RMS(x) w||_2 >= (1/r) ||w_perp||_2
```

Using `r <= r_max` and `||w_perp|| >= beta ||w||` gives the result.

QED.

If `eps = 0` and `w` is exactly radial, the lower bound can be zero. This is
why A1 must keep an alignment assumption.

## 4. LayerNorm Jacobian Caveat

LayerNorm subtracts the mean:

```text
mu(x) = (1/d) 1^T x
c(x) = x - mu(x) 1
sigma(x) = sqrt((1/d)||c(x)||_2^2 + eps)
LN(x) = c(x)/sigma(x)
```

Let:

```text
P_mean = I - (1/d) 11^T
```

Then LayerNorm depends only on `P_mean x`. Therefore:

```text
J_LN(x) 1 = 0
```

The all-ones direction is always a null direction. If `eps=0`, the centered
radial direction is also null. With `eps>0`, it is only damped.

Thus, for LayerNorm, a lower bound is valid only on directions with nontrivial
component in the centered non-radial subspace.

## 5. Theorem A5: Normalization-Safe Residual Gradient

Let the residual value-path vector before projection back to anchor `h_i` be:

```text
w_T = J_N(u_T)^T g_T
```

Assume:

1. Normalization is locally upper-bounded:

```text
||J_N(u_T)||_op <= L_N
```

2. The useful backpropagated gradient is not in the normalization nullspace.
For RMSNorm, a sufficient condition is:

```text
r(u_T) <= r_max
||g_{T,perp}||_2 >= beta ||g_T||_2
```

For LayerNorm, replace `perp` by the centered non-radial subspace.

3. Projection alignment holds:

```text
|<V_a^T w_T, u_i>| >= c_align
```

Then A1's residual value-path lower bound is valid:

```text
|<g_i_value, u_i>|
>= rho_min epsilon c_align
```

and the normalization step contributes no `gamma^(T-i)` factor.

**Proof**

The residual value path from A1 is:

```text
g_i_value = rho_T p_{T,i} V_a^T J_N(u_T)^T g_T
```

The normalization Jacobian is evaluated locally at `u_T`; it does not contain a
RetNet recurrent transition from `i` to `T`. Under the non-nullspace and
alignment assumptions:

```text
|<V_a^T J_N(u_T)^T g_T, u_i>| >= c_align
```

Using:

```text
rho_T >= rho_min
p_{T,i} >= epsilon
```

gives:

```text
|<g_i_value, u_i>|
>= rho_min epsilon c_align
```

QED.

## 6. Consequence For The Main Proof

The normalization condition should be listed explicitly in the global theorem:

```text
bounded activation scale
non-null useful gradient direction
projection alignment after normalization
```

If these fail, the sparse residual cache may contain the right anchor and assign
it enough attention mass, yet still transmit little or no useful gradient.

## 7. Empirical Measurement Contract

Log:

```text
r(u_T) or sigma(u_T)
estimated ||J_N||_op
fraction of gradient energy outside normalization nullspace
|<V_a^T J_N^T g_T, u_i>|
```

For LayerNorm-like maps, also log the gradient energy in:

```text
span(1)
centered radial direction
centered non-radial subspace
```

Only the last part supports a robust lower-bound claim.
