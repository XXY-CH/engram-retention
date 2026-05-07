# Formal Proof C3: Engram Slot Update And Session-Cache Stability

Created: 2026-05-03

Status: first stability theorem for the speculative parts of the Gemini
discussion: online Engram updates, direct slot edits, and temporary session
cache branches.

## 0. Claim Boundary

The Engram paper supports deterministic hashed N-gram retrieval, context-aware
gating, residual fusion, and host-memory offload/prefetch. It does not prove
that arbitrary online updates or direct hash-slot edits are safe.

This note proves a conditional safety statement:

```text
small bounded Engram updates are safe if their collateral collision effect,
residual scale, downstream Lipschitz constant, and task margin satisfy an
explicit inequality.
```

It does not prove that an update is useful or factually correct.

## 1. Objects And Notation

Use a simplified multi-head Engram read:

```text
E(x) = (1/K) sum_{k=1}^K s_k(x) E_k[phi_k(x)]
```

where:

- `x` is a compressed N-gram key;
- `phi_k` is a pairwise-independent hash into `M` slots;
- `s_k(x) in {-1,+1}` is an optional signed-hashing sign;
- `K` is the number of hash heads;
- `E_k[j] in R^d` is the embedding vector in head `k`, slot `j`.

An update writes a bounded vector `u_{x,k}` into each slot touched by target key
`x`:

```text
E'_k[phi_k(x)] = E_k[phi_k(x)] + u_{x,k}
```

Assume:

```text
||u_{x,k}||_2 <= U
```

For a different query key `y != x`, define the collateral update:

```text
xi_y = E'(y) - E(y)
     = (1/K) sum_{k=1}^K s_k(y) 1[phi_k(y)=phi_k(x)] u_{x,k}
```

For `S` independently updated target keys `x_1,...,x_S`, the collateral update
is:

```text
Xi_y = sum_{r=1}^S xi_{y,r}
```

## 2. Lemma C3.1: Single-Update Collateral Probability

**Statement**

For any non-target query key `y != x`, the probability that at least one head is
affected by the update is:

```text
Pr(exists k: phi_k(y)=phi_k(x))
<= K/M
```

**Proof**

For each head:

```text
Pr(phi_k(y)=phi_k(x)) = 1/M
```

by pairwise-independent uniform hashing. Apply the union bound over `K` heads:

```text
Pr(exists collision) <= sum_k 1/M = K/M
```

QED.

This bound is useful for exact hit/collision reasoning, but it does not yet
measure vector magnitude.

## 3. Lemma C3.2: Conservative Collateral Magnitude Bound

**Statement**

Without using signed cancellation, the expected squared collateral magnitude for
one update satisfies:

```text
E[||xi_y||_2^2] <= U^2 / M
```

**Proof**

By convexity of squared norm:

```text
|| (1/K) sum_k a_k ||_2^2
<= (1/K) sum_k ||a_k||_2^2
```

where:

```text
a_k = s_k(y) 1[phi_k(y)=phi_k(x)] u_{x,k}
```

Therefore:

```text
E[||xi_y||_2^2]
<= (1/K) sum_k E[1[phi_k(y)=phi_k(x)] ||u_{x,k}||_2^2]
<= (1/K) sum_k (1/M) U^2
= U^2/M
```

QED.

This is the safe bound when signs or cross-head independence are not trusted.

## 4. Lemma C3.3: Signed-Cancellation Collateral Bound

**Statement**

If signed hashing is used and the cross terms have zero expectation, then:

```text
E[||xi_y||_2^2] <= U^2 / (K M)
```

**Proof Sketch**

Expand:

```text
E[||xi_y||_2^2]
= E[ || (1/K) sum_k a_k ||_2^2 ]
= (1/K^2) sum_k E[||a_k||_2^2]
  + (1/K^2) sum_{k != l} E[<a_k,a_l>]
```

The signed-hashing assumption gives:

```text
E[<a_k,a_l>] = 0
```

For the diagonal terms:

```text
E[||a_k||_2^2] <= U^2/M
```

Thus:

```text
E[||xi_y||_2^2]
<= (1/K^2) K U^2/M
= U^2/(K M)
```

QED.

This stronger version matches the same signed-noise logic used in C1.

## 5. Lemma C3.4: Many Updates

**Statement**

For `S` updated target keys, if the signed cross terms vanish across updates,
then:

```text
E[||Xi_y||_2^2] <= S U^2 / (K M)
```

Without signed cancellation, the conservative sufficient bound is:

```text
E[||Xi_y||_2^2] <= S^2 U^2 / M
```

**Proof**

For the signed version, apply Lemma C3.3 to each update and use the zero-mean
cross-term assumption across updates, so second moments add linearly.

For the conservative version, do not assume cross-update cancellation:

```text
||sum_{r=1}^S xi_{y,r}||_2^2
<= S sum_{r=1}^S ||xi_{y,r}||_2^2
```

Taking expectations and applying Lemma C3.2 gives:

```text
E[||Xi_y||_2^2]
<= S sum_{r=1}^S U^2/M
= S^2 U^2/M
```

QED.

## 6. Theorem C3: Margin-Safe Engram Update

Consider residual injection:

```text
h_clean = h + alpha E(y)
h_edit  = h + alpha E'(y)
```

For a non-target query `y`, the hidden-state perturbation caused by collateral
slot edits is:

```text
h_edit - h_clean = alpha Xi_y
```

Let downstream computation `G` be locally Lipschitz:

```text
||G(u)-G(v)||_2 <= L_G ||u-v||_2
```

Suppose the clean decision has margin `mu > 0`, meaning any output perturbation
below `mu` cannot change the decision.

Using the signed bound, a sufficient condition for preserving the non-target
decision with probability at least `1-delta` is:

```text
L_G |alpha| U sqrt(S/(K M delta)) < mu
```

Equivalently:

```text
K M > (L_G^2 alpha^2 U^2 S)/(mu^2 delta)
```

Without signed cancellation, use the quadratic-in-`S` conservative condition:

```text
M > (L_G^2 alpha^2 U^2 S^2)/(mu^2 delta)
```

**Proof**

From Lemma C3.4 and Lipschitzness:

```text
E[||G(h_edit)-G(h_clean)||_2^2]
<= L_G^2 alpha^2 E[||Xi_y||_2^2]
<= L_G^2 alpha^2 S U^2/(K M)
```

Apply Markov's inequality to the squared output perturbation:

```text
Pr(
  ||G(h_edit)-G(h_clean)||_2
  >= L_G |alpha| U sqrt(S/(K M delta))
) <= delta
```

If this high-probability perturbation radius is below margin `mu`, the decision
is preserved.

QED.

For the no-signed-cancellation version, the same Markov step applies after
replacing Lemma C3.4's signed second-moment bound with:

```text
E[||Xi_y||_2^2] <= S^2 U^2/M
```

## 7. Corollary C3.1: Session Cache Must Be Namespaced Or It Pollutes Global Memory

If session-cache keys share the same hash namespace as global Engram keys, then
`S` session updates contribute directly to the collateral bound for unrelated
global keys:

```text
E[||Xi_y||_2^2] <= S U^2/(K M)
```

This is the signed-cancellation version. Without signed cancellation, use the
more conservative:

```text
E[||Xi_y||_2^2] <= S^2 U^2/M
```

Therefore a separate namespace salt is not cosmetic. It makes the effective
collision set for global keys:

```text
S_global_only
```

instead of:

```text
S_global_only + S_session
```

For multimodal caches, namespace must include at least:

```text
modality_id, user/session id or scope id, n-gram/order id, and table branch id.
```

Otherwise visual/audio/text keys can collide in the same slot space and the
capacity condition must pay for the combined active key count.

## 8. Corollary C3.2: RetNet Temporal Stability After Engram Injection

Let a collateral Engram perturbation `delta h_t = alpha Xi_y` enter a RetNet
layer whose recurrent state satisfies:

```text
S_tau = Gamma S_{tau-1} + B h_tau
||Gamma||_op <= gamma < 1
||B||_op <= L_B
```

Then its effect on a later state is bounded by:

```text
||delta S_tau||_2
<= gamma^(tau-t) L_B ||delta h_t||_2
```

and the total future state-energy contribution is bounded by:

```text
sum_{tau=t}^infty ||delta S_tau||_2
<= (L_B/(1-gamma)) ||delta h_t||_2
```

This shows that RetNet's contractive recurrence does not amplify a bounded
Engram update without limit. The price is the factor:

```text
1/(1-gamma)
```

which can be large when `gamma` is very close to one.

## 9. Consequences For The Architecture

The Gemini-style idea:

```text
freeze backbone, update Engram, keep knowledge current
```

is only defensible under a restricted update policy:

```text
bounded update norm U
bounded active update count S per namespace
small residual scale alpha or adaptive gate
sufficient hash capacity K M
validated downstream margin mu
separate namespaces for global, session, and modality-specific tables
```

This turns "online Engram knowledge update" from a blanket claim into a
testable contract.

## 10. Empirical Checks Required

To validate this theorem in experiments, log:

```text
||u_{x,k}||_2
number of active updates S per namespace
collateral hit rate for unrelated keys
||alpha Xi_y||_2
downstream margin mu
decision flip rate on non-target probes
RetNet future-state perturbation norm over distance
```

An update should be rejected or down-scaled when:

```text
L_G |alpha| U sqrt(S/(K M delta)) >= mu
```

under the signed-cancellation assumptions, or:

```text
L_G |alpha| U S / sqrt(M delta) >= mu
```

without those assumptions. It should also be rejected when non-target flip rate
rises above the tolerated threshold.
