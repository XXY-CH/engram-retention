# Formal Proof A4: Anchor Success Probability Factorization

Created: 2026-05-03

Status: factorizes the `p_A` term used by the global conditional theorem.

Correction status: optional A-time proof for token-time Sparse Anchor
Residuals. It is not a required theorem for Kimi depth-wise AttnRes.

## 0. Claim Boundary

G1 uses:

```text
Pr(A_i) >= p_A
```

where `A_i` is the event that a critical token obtains a non-decaying sparse
attention-residual gradient path. This note makes `p_A` measurable rather than
leaving it as a black-box constant.

## 1. Events

For critical token `i` and prediction position `T`, define:

```text
S_i = {a_i = 1}
```

The critical token is selected into the sparse residual cache.

```text
R_i = {p_{T,i} >= epsilon}
```

The later residual read assigns the cached critical anchor enough attention
mass.

```text
D_i = {residual value path dominates cancellation terms}
```

The non-decaying residual value path is not cancelled by key-path, gate-path, or
other total-gradient terms.

Then:

```text
A_i = S_i ∩ R_i ∩ D_i
```

## 2. Theorem A4: Conditional Product Lower Bound

**Statement**

Assume:

```text
Pr(S_i) >= r
```

```text
Pr(R_i | S_i) >= q
```

```text
Pr(D_i | S_i ∩ R_i) >= d
```

Then:

```text
Pr(A_i) >= r q d
```

Therefore the global theorem can use:

```text
p_A = r q d
```

as a measurable lower bound.

**Proof**

By the probability chain rule:

```text
Pr(A_i)
= Pr(S_i ∩ R_i ∩ D_i)
= Pr(S_i) Pr(R_i | S_i) Pr(D_i | S_i ∩ R_i)
```

Apply the three lower bounds:

```text
Pr(A_i) >= r q d
```

QED.

## 3. Relation To Earlier Proofs

The factors correspond to previous proof obligations:

```text
r = critical-token recall of the sparse gate
q = readout-mass success probability
d = total-gradient dominance success probability
```

A3 gives a sufficient deterministic condition for `R_i`:

```text
s_i >= s_j + Delta for all j != i
```

with:

```text
Delta >= log((m-1)epsilon/(1-epsilon))
```

A2 gives the deterministic dominance condition for `D_i`:

```text
A > C_K + C_Q + C_R gamma^(T-i)
```

where `A` is the aligned residual value-path magnitude and the `C_*` terms bound
competing/cancelling paths.

B2/B3 give sufficient conditions for stable gate selection, but they do not
automatically imply high `r`. High recall for critical tokens remains an
empirical or training-distribution assumption.

## 4. Consequence For G1

Substitute:

```text
p_A = r q d
```

into the G1 lower bound:

```text
Pr(joint success) >= r q d - |Y|delta_C - delta_B
```

Thus the architecture claim is meaningful only if:

```text
r q d > |Y|delta_C + delta_B
```

This is stricter and more useful than saying "the gate should work". It says
the product of:

```text
gate recall × readout success × dominance success
```

must exceed the Engram and gate failure budget.

## 5. Empirical Measurement Contract

For planted-critical-token tasks, estimate:

```text
r_hat = selected_critical_tokens / total_critical_tokens
q_hat = selected_and_read_critical_tokens / selected_critical_tokens
d_hat = noncancelled_directional_successes / selected_and_read_critical_tokens
```

and compare:

```text
r_hat q_hat d_hat
```

against:

```text
|Y|delta_C_hat + delta_B_hat
```

If the left side is not larger, the current proof chain cannot support a
long-range reasoning claim for that experimental regime.
