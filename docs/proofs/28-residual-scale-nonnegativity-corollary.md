# Proof: Non-Negative Residual Scale Preserves Branch Contribution

Created: 2026-05-10

Status: corollary to proof 17, motivated by empirical finding.

## 0. Context

The three residual branches (AttnRes, Engram, Snapshot) each use a trainable
scalar `residual_scale` initialized to `init_scale > 0`. During training,
AdamW with weight decay can push small parameters through zero, producing
negative effective scales. Empirically observed: `attnres_scale = -0.002`
at layer 7 after 500 steps on the needle task.

Fix applied: `effective_scale = residual_scale.abs()` in all forward passes.

## 1. Claim

For any residual branch with contribution:

```text
r = s * f(x)
```

where `s` is the scale parameter and `f(x)` is the branch output,
applying `|s|` ensures:

1. The branch contribution is always in the same direction as the learned
   function output (never subtracts when it should add).
2. The gradient through `|s|` at positive `s` is identity, so training
   dynamics are unchanged for positive scales.
3. The non-dilution theorems (proofs 17, 22) remain valid because their
   lower bounds assume non-negative effective contributions.

## 2. Proof

### 2.1 Sign Preservation

For `s > 0`: `|s| = s`, contribution is `s * f(x)`. Unchanged.

For `s < 0` (the bug case): without abs(), contribution is `s * f(x)`,
which is the *negation* of the intended branch output. With abs(),
contribution is `|s| * f(x)`, restoring the intended sign.

### 2.2 Gradient Preservation

The derivative of `|s|` with respect to `s` is:

```text
d|s|/ds = sign(s) = +1 for s > 0
```

Since `init_scale = 1e-4 > 0` and the parameter is initialized positive,
the gradient is exactly `+1` at initialization and remains `+1` as long as
the parameter stays positive. Training dynamics are identical until the
(unwanted) crossing through zero.

### 2.3 Non-Dilution Preservation

Theorem A-Depth (proof 17) requires `alpha_{i->l} >= epsilon > 0`
and derives a lower bound on directional gradient:

```text
|<dL/dv_i, u_i>| >= epsilon * c
```

This bound depends on the attention weight `alpha` being non-negative.
The residual scale `s` multiplies the *output* of the AttnRes branch:
if `s < 0`, the effective contribution reverses sign, which means the
direction `u_i` that was useful becomes actively harmful. By constraining
`|s|`, we maintain:

```text
effective_contribution = |s| * (alpha_{i->l} * v_i) >= 0
```

which preserves the non-dilution lower bound.

QED.

## 3. What This Does Not Prove

This does not prove that abs() is optimal — a smoother parameterization
like softplus or exp could provide better gradient properties near zero.
It proves only that abs() prevents the regression observed empirically
without changing the training dynamics for the normal operating regime
(s > 0).

## 4. Alternative Parameterizations Considered

| Method | Pros | Cons |
|--------|------|------|
| `abs()` | Exact init preservation, simplest | Non-smooth at 0 |
| `softplus(s)` | Smooth everywhere | Changes init: softplus(1e-4) ≈ 0.693 |
| `exp(s)` | Strictly positive | Changes init: exp(1e-4) ≈ 1.0001 |
| `squared(s)` | Smooth, positive | Changes gradient: 2s at init |

`abs()` was chosen because: (a) it preserves init exactly, (b) the
non-smooth point at s=0 is never reached in practice (the parameter
would need to cross through zero, which triggers the fix), and (c)
minimal code change.
