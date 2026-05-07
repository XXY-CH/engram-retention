# Proof Closure: Conditional Feasibility Of Earlier Sparse-Anchor Formulation

Created: 2026-05-03

Status: closure of the earlier mathematical proof chain. After reading Kimi
Attention Residuals and deferring MoE from the active phase-1 baseline, this remains
valid only for the optional token-time Sparse Anchor Residual. It is not the
final proof closure for the active Dense RetNet + Block AttnRes + Engram
architecture.

Update note:

```text
Kimi Block AttnRes is depth-axis attention over layer/block outputs.
The theorem below is token-time sparse-anchor logic.
Do not cite it as the proof of Kimi AttnRes without translating T-i to L-l and
replacing token anchors with layer/block residual sources.
```

## 0. Final Claim Type

The result is a conditional feasibility theorem, not an unconditional model
success theorem.

We can now rigorously claim:

```text
Under explicit contractivity, selection, readout, normalization, dominance,
hash-capacity, update-stability, and budget-margin assumptions, the hybrid
architecture has a non-exponentially-decaying sparse residual path for critical
tokens while keeping Engram retrieval/update perturbations below task margins.
```

We cannot claim:

```text
the gate will always discover critical tokens
Engram online updates are automatically safe
the whole system is O(1) in long-context length
RetNet alone preserves arbitrary long-range reasoning state
```

## 1. Architecture Being Proven

The earlier target architecture was:

```text
RetNet backbone
  = O(1) recurrent streaming compute per token

Sparse Anchor Residual
  = O(B) long-range anchor cache for rare critical reasoning tokens

Engram
  = O(K) deterministic hashed memory lookup with offloadable tables
```

For the realigned architecture, add:

```text
Block Attention Residuals
  = O(N_depth) depth-axis block summaries

Dense MLP/FFN
  = ordinary per-token channel mixing
```

Therefore the older resource claim is:

```text
O(1) + O(K) + O(B)
```

per token, where `K` and `B` must be controlled by design.

## 2. Required Assumptions

### A-side: RetNet and sparse residual

```text
||Gamma||_op <= gamma < 1
```

RetNet recurrence is contractive.

```text
a_i = 1
p_{T,i} >= epsilon
```

The critical token is written and later read with sufficient attention mass.

```text
A > C_K + C_Q + C_R gamma^(T-i)
```

The sparse residual value path dominates key/gate/other cancellation terms.

```text
normalization useful direction is outside null or near-null subspace
```

LayerNorm/RMSNorm does not erase the useful residual gradient.

### C-side: Engram retrieval and update

For retrieval collision noise:

```text
E[||eta||_2^2] <= ((N-1)R^2)/(K M)
```

and a sufficient margin condition:

```text
L_G |alpha| R sqrt((N-1)/(K M delta_C1)) < mu/2
```

For signed online/session updates:

```text
L_G |alpha| U sqrt(S/(K M delta_C3)) < mu/2
```

For conservative no-signed-cancellation updates:

```text
L_G |alpha| U S / sqrt(M delta_C3) < mu/2
```

Session and multimodal caches require separate namespaces.

### B-side: sparse gate and budget

Oracle threshold:

```text
a_i*(lambda)=1[Delta_i > lambda]
```

Hard budget:

```text
select top positive utilities up to B
```

Estimated utilities are stable only if:

```text
Delta_(B) - Delta_(B+1) > 2 eta_Delta
```

Adaptive memory price is stable only if:

```text
eta_price n < Delta_(B) - Delta_(B+1) - 2 eta_Delta
```

## 3. Main Closed Theorem

Let:

```text
r = Pr(critical token is selected)
q = Pr(readout mass >= epsilon | selected)
d = Pr(residual value path dominates cancellation | selected and read)
```

Let:

```text
delta_C = Engram margin-failure probability per protected probe
delta_B = gate/budget instability probability
Y       = finite protected non-target probe set
```

Then the joint success probability is bounded below by:

```text
Pr(joint success)
>= r q d - |Y|delta_C - delta_B
```

On the joint success event, for a critical token `i` and later position `T`,
there exists a unit direction `v_i` and constant `g_min > 0`, independent of
`T-i`, such that:

```text
<grad_{h_i} L_T, v_i>_+ >= g_min
```

while the RetNet-only recurrent path satisfies:

```text
||(grad_{h_i} L_T)_RetNet|| <= C_R gamma^(T-i)
```

Thus the sparse residual path removes the exponential distance factor only on
successful selected/read/dominant events.

## 4. Proof Sketch From Existing Artifacts

1. A1 proves the RetNet path contains `Gamma^(T-i)` but the sparse residual
   value path does not.
2. A2 proves that total-gradient non-decay requires dominance over cancellation
   terms.
3. A3 proves readout mass follows from a softmax logit margin.
4. A4 factorizes residual-path success probability into `r q d`.
5. A5 proves normalization is safe only under non-nullspace and bounded-scale
   assumptions.
6. C1 proves hash collision and signed-noise variance bounds.
7. C2 converts Engram retrieval noise into a downstream margin-safe condition.
8. C3 extends the margin-safe condition to slot edits and session-cache updates.
9. B1/B2 prove threshold and top-budget oracle gate forms.
10. B3 proves utility-estimation robustness under margin gap.
11. B4 proves adaptive memory-price stability under boundary gap and small step.
12. G1 composes these events with union bound.

## 5. What Would Falsify The Claim

The proof chain fails in any regime where:

```text
r q d <= |Y|delta_C + delta_B
```

or where any of the structural assumptions fail:

```text
gamma >= 1
critical anchors are not selected
readout mass collapses as cache grows
residual value path cancels in total gradient
normalization kills the useful direction
Engram collision/update noise exceeds task margin
budget boundary gap is too small for utility-estimation error
adaptive memory-price step jumps across the stable interval
```

These are not minor details. They are the actual load-bearing conditions.

## 6. Current Proof Status

Within the simplified token-time sparse-anchor model, the proof chain is closed:

```text
A + C + B => G1 => closed conditional feasibility theorem
```

The next phase should not keep adding unbounded theory before measurement. It
should build synthetic tests that estimate:

```text
r, q, d, delta_C, delta_B, gamma, epsilon, B, K, M
```

and check the inequality:

```text
r q d > |Y|delta_C + delta_B
```

If that inequality holds on controlled tasks, the optional Sparse Anchor
Residual has empirical support for the mathematically proven regime. If it does
not hold, the proof itself tells us which module failed.

The active main architecture now requires a separate depth-axis AttnRes proof,
Engram safety proof, and Dense RetNet implementation check. It does not require
MoE routing or expert-capacity assumptions.
