# A. RetNet Decay And Sparse Residual Gradient Interference

Created: 2026-05-03

Correction status: historical A-time proof for the optional Sparse Anchor
Residual. This is not the proof of Kimi Attention Residuals. Kimi AttnRes is
depth-axis attention over previous layer/block outputs; see
`docs/proofs/17-depth-attnres-non-dilution-proof.md` for the active A-depth
proof.

## Goal

Prove the first viable claim:

> If a critical early token is written into a sparse attention-residual cache and
> later receives non-negligible attention mass, then the loss gradient at a far
> future token can reach that early token through a path that does not contain
> the RetNet exponential decay factor.

This is a conditional anti-decay theorem. It does not say the gate will always
select the right token.

## Minimal Model

Let a single RetNet-style recurrent state be:

```text
S_t = Gamma S_{t-1} + k_t v_t^T
```

with spectral radius:

```text
rho(Gamma) <= gamma < 1
```

The retention readout is:

```text
r_t = q_t S_t
```

Add a sparse attention-residual cache. At token `i`, the gate writes an anchor:

```text
a_i in {0, 1}
c_i = (K_a h_i, V_a h_i)
```

The cache at time `T` contains selected anchors:

```text
C_T = {c_j : j <= T, a_j = 1}
```

The residual readout is:

```text
z_T = Attn(Q_a h_T, C_T)
    = sum_{j: a_j=1} p_{T,j} V_a h_j
```

where:

```text
p_{T,j} = softmax_j((Q_a h_T)^T K_a h_j / sqrt(d_a))
```

A simplified hidden update near `T` is:

```text
h_T^+ = N(h_T + F(r_T) + rho_T z_T)
```

where `N` is LayerNorm/RMSNorm or an abstract normalization map.

## Gradient Decomposition

For an early token `i`, the total derivative from `L_T` to `h_i` has at least
three conceptual parts:

```text
dL_T/dh_i
= (dL_T/dh_i)_RetNet
+ (dL_T/dh_i)_residual-cache
+ (dL_T/dh_i)_gate
```

The RetNet-only path repeatedly passes through `Gamma`, so under bounded local
Jacobians:

```text
||(dL_T/dh_i)_RetNet|| <= C_R gamma^(T-i)
```

This is the expected long-range decay.

If `a_i = 1`, the value path contributes:

```text
(dL_T/dh_i)_value
= rho_T (dL_T/dh_T^+) J_N p_{T,i} V_a
```

This term does not include `Gamma^(T-i)`. Its magnitude is controlled by
attention mass, residual scale, projection norms, and normalization Jacobian.

There is also a key path because `h_i` influences the attention score through
`K_a h_i`:

```text
(dL_T/dh_i)_key
= rho_T (dL_T/dz_T) (dz_T/dp_T) (dp_T/d(K_a h_i)) K_a
```

The key path can help or hurt depending on the softmax geometry. For the first
proof, the value path is enough to establish a non-decaying route when
`p_{T,i}` is bounded below.

## Sufficient Conditions

Assume:

```text
||dL_T/dh_T^+|| >= ell_min
||J_N u|| >= n_min ||u|| on the residual subspace
sigma_min(V_a on useful subspace) >= v_min
rho_T >= rho_min > 0
p_{T,i} >= epsilon > 0
a_i = 1
```

Then the value path has lower bound:

```text
||(dL_T/dh_i)_value||
>= rho_min * ell_min * n_min * epsilon * v_min
```

up to alignment/cancellation constants. A fully rigorous version should express
this as an inner-product lower bound against a useful direction `u_i`, because
vector norm lower bounds can fail under adversarial cancellation.

## Theorem Skeleton

**Theorem A1: Conditional Non-Decaying Gradient Path**

For the simplified RetNet plus sparse attention-residual model above, suppose
all local Jacobians are bounded above, and suppose the residual value path from
`h_i` to `L_T` has a nonzero directional alignment lower bound `c_align > 0`.
If `a_i = 1` and `p_{T,i} >= epsilon`, then:

```text
|<dL_T/dh_i, u_i>|
>= c_align * rho_min * epsilon - C_R gamma^(T-i)
```

for a useful direction `u_i`. Therefore, as `T-i -> infinity`, the RetNet path
vanishes but the residual-cache path remains bounded away from zero whenever
the selection and readout conditions hold.

## Immediate Corollary

If the gate is stochastic and:

```text
Pr(a_i = 1 | i is critical) >= r
Pr(p_{T,i} >= epsilon | a_i = 1, i is critical) >= s
```

then a non-decaying gradient path exists with probability at least `r * s`.
The expected directional signal is lower-bounded only after accounting for
possible sign cancellation; this requires either aligned objectives, contrastive
anchor supervision, or an empirical gradient-sign diagnostic.

## What This Does And Does Not Prove

Proved by this route:

- RetNet decay alone loses far-past gradient signal under `rho(Gamma)<1`.
- A sparse residual value path can bypass the decay factor.
- The real bottleneck is gate recall and later attention mass, not distance.

Not proved yet:

- that the model learns `a_i = 1` for critical tokens;
- that `p_{T,i}` stays above `epsilon` under large cache size;
- that normalization never projects away the residual path;
- that key-path gradients are stable;
- that this holds for the current repository implementation, which does not yet
  implement sparse dynamic KV anchors.

## Next Proof Tasks

1. Replace the informal norm lower bound with a directional derivative theorem.
   First formal draft completed in `docs/proofs/02-conditional-nondecay-gradient-proof.md`.
2. Add cache-size dependence: if `m = |C_T|`, random unstructured attention gives
   `p_{T,i} ~= 1/m`, so useful lower bounds require learned routing or anchor
   salience.
3. Connect A to B by defining token utility:

```text
Delta_i = L_without_anchor_i - L_with_anchor_i
```

4. Connect A to implementation by adding a minimal sparse-anchor layer and a
   synthetic copy/retrieval task that logs `a_i`, `p_{T,i}`, and gradient norms.
