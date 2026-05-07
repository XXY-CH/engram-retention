# Formal Proof A-Depth: Block AttnRes Gives A Direct Depth-Wise Path

Created: 2026-05-03

Status: first depth-axis proof corresponding to Kimi Attention Residuals.

## 0. Claim Boundary

This proof is about Kimi-style Attention Residuals:

```text
depth-axis attention over previous layer/block outputs
```

It is not the token-time Sparse Anchor Residual proof. The two have similar
mathematical shape but different axes.

## 1. Standard Residual Accumulation

A simplified residual stack writes:

```text
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

Unrolling gives:

```text
h_l = h_1 + sum_{i=1}^{l-1} f_i(h_i)
```

This gives an identity gradient highway, but all earlier layer outputs are
aggregated with fixed unit weights. As Kimi AttnRes notes, this can cause
depth-wise hidden-state growth and contribution dilution.

The issue is not that gradients cannot flow at all. The issue is:

```text
deeper layers cannot selectively recover a specific earlier layer/block source
after fixed aggregation has mixed all sources into one state.
```

## 2. Full Attention Residuals

Kimi AttnRes defines:

```text
h_l = alpha_{0->l} h_1 + sum_{i=1}^{l-1} alpha_{i->l} f_i(h_i)
```

where:

```text
sum_{i=0}^{l-1} alpha_{i->l} = 1
alpha_{i->l} = softmax_i(score_{i,l})
```

Let:

```text
v_i = f_i(h_i)
```

for ordinary layer outputs, and `v_0 = h_1` for the embedding/source state.

Then:

```text
h_l = sum_{i=0}^{l-1} alpha_{i->l} v_i
```

## 3. Lemma A-D1: Selected Depth Source Has A Direct Value Path

**Statement**

Condition on a previous source `v_i` receiving depth-attention mass:

```text
alpha_{i->l} >= epsilon_depth > 0
```

Treating attention weights as fixed for the value-path derivative:

```text
partial h_l / partial v_i = alpha_{i->l} I
```

This derivative does not include a product of intermediate layer Jacobians from
`i+1` to `l-1`.

**Proof**

From:

```text
h_l = sum_j alpha_{j->l} v_j
```

holding `alpha` fixed:

```text
partial h_l / partial v_i = alpha_{i->l} I
```

No term of the form:

```text
prod_{k=i+1}^{l-1} partial h_k / partial h_{k-1}
```

appears in this path. Therefore AttnRes creates direct selected access from
layer/block source `i` to layer `l`.

QED.

## 4. Lemma A-D2: Depth Softmax Margin Gives Readout Mass

If source `i` has score margin:

```text
s_i >= s_j + Delta_depth
```

for all `j != i` among `m_depth` sources, then:

```text
alpha_{i->l} >= 1 / (1 + (m_depth - 1) exp(-Delta_depth))
```

Thus a sufficient condition for:

```text
alpha_{i->l} >= epsilon_depth
```

is:

```text
Delta_depth >= log((m_depth - 1) epsilon_depth / (1 - epsilon_depth))
```

This is exactly the depth-axis analogue of the token-time softmax mass proof.

## 5. Theorem A-Depth: Conditional Non-Diluted Depth Signal

Let `G_l` be the downstream loss from layer `l` onward, and let:

```text
g_l = dL / dh_l
```

Assume:

```text
alpha_{i->l} >= epsilon_depth
|<g_l, u_i>| >= c_depth
```

for some useful unit direction `u_i` in the source representation space.

Then the value-path directional gradient to `v_i` satisfies:

```text
|<dL/dv_i, u_i>|
>= epsilon_depth c_depth
```

and this lower bound does not depend exponentially on depth distance `l-i`.

**Proof**

From Lemma A-D1:

```text
dL/dv_i = alpha_{i->l} g_l
```

on the fixed-attention value path. Taking the directional inner product:

```text
|<dL/dv_i, u_i>|
= alpha_{i->l} |<g_l, u_i>|
>= epsilon_depth c_depth
```

No product over intermediate depth Jacobians appears in this path.

QED.

## 6. Block AttnRes Version

For Block AttnRes, replace individual layer source `v_i` by block source `b_n`:

```text
h_l = sum_{n=0}^{N_depth} beta_{n->l} b_n
```

If:

```text
beta_{n->l} >= epsilon_depth
```

then the same proof gives:

```text
|<dL/db_n, u_n>| >= epsilon_depth c_depth
```

The resource count is controlled by the number of block summaries:

```text
N_depth
```

not by the total number of layers `L`.

## 7. What This Does And Does Not Prove

It proves:

```text
Kimi-style AttnRes can provide direct depth-wise access to earlier layer/block
sources when attention mass and alignment conditions hold.
```

It does not prove:

```text
token-time exact factual recall over long context
```

That remains either RetNet/gated-retention's job or the optional Budgeted Sparse
Attention Anchor module.
