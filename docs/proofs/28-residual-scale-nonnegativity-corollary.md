# Corollary: Non-Negative Residual Scale Prevents Sign Reversal

Created: 2026-05-10

Status: engineering corollary to conditional value-path proofs, motivated by an
empirical sign-regression finding.

## 0. Context

The three residual branches (AttnRes, Engram, Snapshot) each use a trainable
scalar `residual_scale` initialized to `init_scale > 0`. During training,
AdamW with weight decay can push small parameters through zero, producing
negative effective scales. Empirically observed: `attnres_scale = -0.002`
at layer 7 after 500 steps on the needle task.

Fix applied: `effective_scale = residual_scale.abs()` in all guarded branch
forward passes.

## 1. Claim

For any residual branch with contribution:

```text
r = s * f(x)
```

where `s` is the scale parameter and `f(x)` is the branch output,
applying `|s|` ensures:

1. The branch scale cannot reverse the sign of the learned branch output.
2. The gradient through `|s|` at positive `s` is identity, so training
   dynamics are unchanged for positive scales.
3. The earlier conditional value-path theorems can continue to use a
   non-negative branch-scale assumption, provided their alignment and margin
   assumptions also hold.

## 2. Proof

### 2.1 Sign Preservation

For `s > 0`: `|s| = s`, contribution is `s * f(x)`. Unchanged.

For `s < 0` (the bug case): without abs(), contribution is `s * f(x)`, which
reverses the branch output. With abs(), contribution is `|s| * f(x)`, so the
scale no longer causes sign reversal. This does not prove that `f(x)` itself is
aligned with the downstream objective; that remains a separate assumption or
empirical measurement.

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

Theorem A-Depth (proof 17) requires `alpha_{i->l} >= epsilon > 0` and a useful
directional alignment condition. It derives a value-path lower bound:

```text
|<dL/dv_i, u_i>| >= epsilon * c
```

This bound depends on the attention weight `alpha` being non-negative and on
the useful direction being preserved. The residual scale `s` multiplies the
output of the AttnRes branch. If `s < 0`, the scale can flip a branch output
that had positive alignment with a downstream direction. By using `|s|`, the
branch-scale factor remains non-negative:

```text
effective_scale = |s| >= 0
```

Therefore `abs()` removes one failure mode for the earlier lower bound: sign
reversal by the scalar gate. It does not, alone, prove the attention-mass,
alignment, or logit-margin assumptions required by those theorems.

QED.

## 3. What This Does Not Prove

This does not prove that `abs()` is optimal. A smoother positive
parameterization such as shifted softplus or exponential reparameterization can
avoid the non-smooth point at zero. This corollary proves only that `abs()`
prevents scalar sign reversal without changing the forward value or derivative
in the normal positive-scale regime (`s > 0`).

## 4. Alternative Parameterizations Considered

| Method | Pros | Cons |
|--------|------|------|
| `abs()` | Exact init preservation, simplest | Non-smooth at 0 |
| `softplus(s)` | Smooth everywhere | Needs inverse-softplus initialization to preserve tiny scale |
| `exp(s)` | Strictly positive | Needs log-scale parameterization to preserve tiny scale |
| `squared(s)` | Smooth, positive | Changes gradient: 2s at init |

`abs()` was chosen because it preserves the current parameter value exactly for
positive scales and is the smallest code change. Its remaining weakness is the
non-smooth point at `s=0`; future experiments should compare it with smooth
positive parameterizations.
