# Residual Injection Composition Guard

Created: 2026-05-03

Status: active design guard for composing Engram and Block AttnRes in the Dense
baseline.

## 0. Why This Exists

Proof 18 deliberately left one local gap:

```text
Dense RetNet block Jacobian / normalization stability with Block AttnRes and
Engram residual injection composed in the same layer stack.
```

This document does not prove a universal stability theorem. It records a
sufficient engineering contract that makes the same-layer interaction small at
initialization and measurable during training.

## 1. Problem

At a layer `l`, a naive composition might do:

```text
x_{l+1} = x_l
        + R_l(N(x_l))
        + E_l(N(x_l))
        + A_l(N(x_l))
        + F_l(N(...))
```

where:

```text
R_l = RetNet/retention branch
E_l = Engram hash-retrieval residual
A_l = Block AttnRes depth-readout residual
F_l = Dense MLP/FFN
N   = RMSNorm/LayerNorm
```

The risk is that `E_l` and `A_l` are both high-variance nonlocal residuals:

```text
E_l: hash-table vector, collision/noise-sensitive
A_l: cross-depth feature readout, possibly from much earlier layers
```

If both enter the same residual stream at large scale, normalization may shrink
or rotate the useful direction from the other branch.

## 2. Active Composition Contract

The first Dense implementation must satisfy at least one placement guard and one
scale guard.

### Placement Guard

Prefer interleaving:

```text
Engram layers:    shallow/sparse, e.g. {2, 6} or {2, 15} as hypotheses
Block AttnRes:    block boundaries, not the same injection point as Engram
```

This is aligned with the Engram paper's layer sensitivity: early Engram
insertion is useful, and splitting memory across selected layers can improve
performance. Exact RetNet layer indices remain a hyperparameter, not a theorem.

### Scale Guard

Each nonlocal residual branch must have an independent small learnable scale:

```text
x_{l+1} = x_l
        + R_l(N(x_l))
        + lambda_E,l E_l(N(x_l))
        + lambda_A,l A_l(N(x_l))
        + F_l(N(x_l))
```

with initialization:

```text
|lambda_E,l| <= eps_scale
|lambda_A,l| <= eps_scale
eps_scale in [1e-5, 1e-4] for the first training attempt
```

Scalar LayerScale is sufficient for the first implementation. Per-channel
LayerScale can be tested later.

### Gate Guard

If a branch has a sigmoid gate, initialize it default-closed:

```text
g = sigmoid(z + b)
b < 0
```

For example:

```text
b = -2 or -3
```

This keeps early training close to the Dense RetNet baseline.

### Parallel-Normalized Branch Guard

Prefer giving all branches the same normalized input:

```text
u_l = RMSNorm(x_l)
x_{l+1} = x_l
        + R_l(u_l)
        + lambda_E,l E_l(u_l)
        + lambda_A,l A_l(u_l)
        + F_l(u_l)
```

Avoid serially feeding a newly injected Engram vector through another
normalization before AttnRes in the same block.

## 3. Sufficient Small-Perturbation Bound

Let the combined nonlocal perturbation be:

```text
p_l = lambda_E,l E_l(u_l) + lambda_A,l A_l(u_l)
```

If:

```text
||E_l(u_l)|| <= B_E
||A_l(u_l)|| <= B_A
|lambda_E,l| B_E + |lambda_A,l| B_A <= tau_l
```

then:

```text
||p_l|| <= tau_l
```

If the downstream normalized Dense block is locally Lipschitz with constant
`L_l`, the perturbation passed into the next block changes the next hidden state
by at most:

```text
O(L_l tau_l)
```

Therefore initialization with small independent scales makes the combined
Engram+AttnRes injection an explicit small perturbation of the Dense RetNet
baseline.

This is only a local bound. During training, `lambda_E,l`, `lambda_A,l`,
`||E_l||`, `||A_l||`, and normalization Jacobian norms must be logged.

## 4. Context Routing Rule

Fresh user-provided context is not Engram memory.

```text
new user context -> RetNet recurrent state + Block AttnRes depth summaries
static/semi-static factual patterns -> Engram
```

Do not online-write fresh documents into the global Engram table unless proof
10's C3 update conditions and namespace isolation are satisfied.

## 5. Implementation Requirements

The first code implementation should include:

```text
lambda_engram parameters, initialized near zero
lambda_attnres parameters, initialized near zero
negative gate bias for Engram fusion gates
configurable Engram insertion layers
configurable Block AttnRes block boundaries
runtime logging for branch norms and lambda values
NaN/Inf checks during warmup
```

## 6. Impact On Proof 18

Proof 18's assumption:

```text
Dense MLP/FFN Jacobians and normalization Jacobians remain bounded and do not
erase the useful direction.
```

is not automatic. This guard turns it into an implementation contract:

```text
bounded branch norms + small independent scales + interleaved placement
```

are the first sufficient conditions to test.
