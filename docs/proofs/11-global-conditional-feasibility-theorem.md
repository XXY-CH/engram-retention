# Global Conditional Feasibility Theorem

Created: 2026-05-03

Status: first composition theorem for the RetNet + sparse attention residual +
Engram architecture.

Correction status: historical optional-anchor composition theorem. It should
not be cited as the active baseline theorem because Kimi AttnRes is depth-axis,
and MoE has been deferred from the current phase-1 baseline. The active composition must
combine Dense RetNet, Block AttnRes, and Engram.

## 0. Claim Boundary

This theorem does not prove that the proposed architecture always works. It
proves a conditional implication:

```text
if sparse anchors are selected/read, Engram noise and updates stay below margin,
and the gate is stable under utility-estimation error, then the hybrid model has
a non-exponentially-decaying long-range path while retaining bounded external
memory risk.
```

The theorem is deliberately probabilistic and pathwise. It avoids the false
claim that the total gradient is always non-decaying.

## 1. Architecture Abstraction

At time `t`, let the hidden update be:

```text
h_{t+1} = N(
  h_t
  + F_ret(h_t, S_t)
  + alpha_t E_hat(x_t)
  + rho_t Z_t(C_t)
)
```

where:

- `S_t` is the RetNet recurrent state;
- `E_hat(x_t)` is Engram retrieval, including hash noise and possible updates;
- `C_t` is the sparse attention-residual anchor cache;
- `Z_t(C_t)` is the residual attention readout from that cache;
- `N` denotes normalization and remaining bounded residual processing.

The resource profile is:

```text
O(1) RetNet recurrent state per token
+ O(K) Engram active hash-head reads per token
+ O(B) sparse anchor cache reads when |C_t| <= B
```

The model is therefore small-memory only when `K` and `B` are controlled.

## 2. Events

For a critical token `i` and prediction position `T`, define these events.

### A-event: Long-range residual path succeeds

```text
A_i = {
  a_i = 1,
  p_{T,i} >= epsilon,
  residual value path dominates cancellation terms
}
```

From A1/A2/A3, on `A_i` there exists a unit direction `v_i` and constant
`g_min > 0` independent of `T-i` such that:

```text
<grad_{h_i} L_T, v_i>_+ >= g_min
```

Here `<.,.>_+ = max(<.,.>, 0)` records the non-decaying positive directional
component. This positive-part form prevents cancellation outside the successful
path from being hidden.

Assume:

```text
Pr(A_i) >= p_A
```

where `p_A` includes gate recall, readout success, and dominance success.
Proof A4 further factorizes this as:

```text
p_A >= r q d
```

under conditional lower bounds on selection, readout, and dominance.

### C-event: Engram retrieval/update is margin-safe

Let `C_y` be the event that Engram hash noise and any slot/session-cache updates
for a non-target key `y` perturb downstream outputs by less than its decision
margin.

From C1/C2/C3, sufficient signed-hashing conditions are:

```text
L_G |alpha| R sqrt((N-1)/(K M delta_C1)) < mu_y/2
```

for ordinary collision noise, and:

```text
L_G |alpha| U sqrt(S/(K M delta_C3)) < mu_y/2
```

for online/session update collateral.

Then:

```text
Pr(C_y) >= 1 - delta_C1 - delta_C3
```

by union bound.

Without signed-cancellation assumptions, replace the update term by:

```text
L_G |alpha| U S / sqrt(M delta_C3) < mu_y/2
```

### B-event: Budgeted gate is stable

Let `B_gate` be the event that the learned/estimated gate selects the same
anchor set as the oracle top-budget rule.

From B2/B3, sufficient conditions are:

```text
Delta_(B) > max(0, Delta_(B+1))
```

and utility estimation error:

```text
|Delta_hat_t - Delta_t| <= eta
```

with:

```text
Delta_(B) - Delta_(B+1) > 2 eta
```

If this holds for the relevant sequence, then:

```text
Pr(B_gate) = 1
```

conditional on the error bound. If the error bound itself only holds with
probability `1-delta_B`, then:

```text
Pr(B_gate) >= 1-delta_B
```

## 3. Theorem G1: Conditional Long-Range Signal With Bounded Memory Risk

**Statement**

Assume:

1. RetNet recurrence is contractive:

```text
||Gamma||_op <= gamma < 1
```

2. Sparse cache size is bounded:

```text
|C_t| <= B
```

3. Normalization on the sparse residual path is locally stable and non-null on
the useful backpropagated direction as specified by A5.

4. For a critical token `i`, the A-event satisfies:

```text
Pr(A_i) >= p_A
```

5. Engram collision/update conditions imply:

```text
Pr(C_y) >= 1 - delta_C
```

for all protected non-target probes `y` in a finite probe set `Y`, where:

```text
delta_C = delta_C1 + delta_C3
```

6. Gate stability holds with probability at least:

```text
1 - delta_B
```

For adaptive memory-price control, B4 additionally requires a positive
top-budget utility gap and a price step small enough not to jump across the
stable interval.

Then:

```text
Pr(
  exists non-decaying positive directional gradient at i
  and all protected Engram probes remain margin-safe
  and the budgeted gate is stable
)
>= p_A - |Y| delta_C - delta_B
```

On that event:

```text
<grad_{h_i} L_T, v_i>_+ >= g_min
```

with `g_min` independent of `T-i`, while the RetNet-only path still decays as:

```text
O(gamma^(T-i))
```

The per-token computational/memory access profile is:

```text
O(1) + O(K) + O(B)
```

not global `O(1)` unless `K` and `B` are treated as constants by design.

## 4. Proof

From A1/A2/A3, event `A_i` implies the existence of a positive directional
gradient lower bound:

```text
<grad_{h_i} L_T, v_i>_+ >= g_min
```

where `g_min` depends on residual scale, value projection, normalization
Jacobian lower bound, attention mass `epsilon`, and cancellation-dominance
margin, but not on `T-i`.

From RetNet contractivity, the RetNet-only recurrent contribution along paths
that must pass through `Gamma` is bounded by:

```text
C gamma^(T-i)
```

so it vanishes with distance. This does not remove the sparse residual path
because the residual cache path:

```text
h_i -> (K_a h_i, V_a h_i) -> Z_T -> h_T -> L_T
```

does not contain `Gamma^(T-i)`.

From C1/C2/C3, each protected Engram probe `y` is margin-safe with probability
at least `1-delta_C`. Applying union bound over finite `Y` gives:

```text
Pr(all y in Y are margin-safe) >= 1 - |Y| delta_C
```

From B3, the budgeted gate is stable with probability at least `1-delta_B`.

Now apply union bound to the complement of the desired joint event:

```text
Pr(A_i and all C_y and B_gate)
>= 1 - Pr(not A_i) - Pr(any not C_y) - Pr(not B_gate)
>= 1 - (1-p_A) - |Y|delta_C - delta_B
= p_A - |Y|delta_C - delta_B
```

On this joint event, all claimed properties hold simultaneously.

QED.

## 5. Interpretation

The theorem makes the project target precise:

```text
The architecture is feasible only if p_A is not too small and the Engram/gate
failure budget |Y|delta_C + delta_B is smaller than p_A.
```

This is the rigorous form of the original intuition:

```text
RetNet handles streaming compute.
Optional Sparse Anchor Residual handles rare long-range reasoning anchors if
that extension is reintroduced.
Engram handles external static/semi-static memory if collision and update
perturbations stay below margin.
```

For the active Dense baseline, this theorem is appendix material. The active
depth-wise AttnRes proof is `docs/proofs/17-depth-attnres-non-dilution-proof.md`.

## 6. Immediate Research Obligations

The theorem exposes the empirical quantities that must be measured:

```text
p_A        = anchor selection/readout/dominance success probability
epsilon    = attention mass on critical anchors
g_min      = directional positive gradient lower bound on success events
delta_C    = Engram collision/update margin failure probability
delta_B    = gate instability probability under utility-estimation error
B          = sparse cache budget
K, M       = Engram hash heads and slots
```

If experiments cannot make:

```text
p_A > |Y|delta_C + delta_B
```

then the current proof chain cannot support the architecture claim.

Using A4, this becomes the measurable condition:

```text
r q d > |Y|delta_C + delta_B
```
