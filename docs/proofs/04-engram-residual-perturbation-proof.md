# Formal Proof C2: Engram Retrieval Noise As Bounded Residual Perturbation

Created: 2026-05-03

Status: first bridge theorem from hash capacity to hidden-state safety.

## 0. Claim Boundary

C1 bounded the retrieval noise of a signed multi-head Engram table:

```text
e_hat = e + eta
E[||eta||_2^2] <= ((N-1) R^2)/(K M)
```

C2 asks what this means after residual injection:

```text
h_plus = h + alpha e_hat
```

This proof does not show that `e` is useful. It only shows when collision noise
is small enough to be treated as a bounded perturbation.

## 1. Objects And Notation

Let:

```text
h_clean = h + alpha e
h_noisy = h + alpha(e + eta)
```

Then:

```text
h_noisy - h_clean = alpha eta
```

Let `G` be the downstream computation from the injected hidden state to either
logits, loss, or a task score:

```text
o_clean = G(h_clean)
o_noisy = G(h_noisy)
```

Assume `G` is locally Lipschitz with constant `L_G`:

```text
||G(u) - G(v)||_2 <= L_G ||u - v||_2
```

This is a local assumption over the activation region under study. It can be
validated empirically with Jacobian-norm estimates.

## 2. Lemma C2.1: Expected Output Error Bound

**Statement**

If C1's noise bound holds, then:

```text
E[||o_noisy - o_clean||_2^2]
<= L_G^2 alpha^2 ((N-1) R^2)/(K M)
```

**Proof**

By Lipschitzness:

```text
||o_noisy - o_clean||_2
<= L_G ||h_noisy - h_clean||_2
= L_G |alpha| ||eta||_2
```

Square and take expectation:

```text
E[||o_noisy - o_clean||_2^2]
<= L_G^2 alpha^2 E[||eta||_2^2]
<= L_G^2 alpha^2 ((N-1) R^2)/(K M)
```

QED.

## 3. Lemma C2.2: High-Probability Error Bound

**Statement**

For any `delta in (0,1)`, Markov's inequality gives:

```text
Pr(
  ||o_noisy - o_clean||_2
  >= L_G |alpha| R sqrt((N-1)/(K M delta))
) <= delta
```

**Proof**

Apply Markov to the nonnegative random variable:

```text
X = ||o_noisy - o_clean||_2^2
```

Using Lemma C2.1:

```text
Pr(X >= E[X]/delta) <= delta
```

Taking square roots yields the bound.

QED.

## 4. Theorem C2: Margin-Safe Engram Injection

Suppose the clean downstream decision has margin `mu > 0`, and changing the
output by less than `mu` cannot change the decision. A sufficient condition for
Engram collision noise to preserve the decision with probability at least
`1-delta` is:

```text
L_G |alpha| R sqrt((N-1)/(K M delta)) < mu
```

Equivalently:

```text
K M > (L_G^2 alpha^2 R^2 (N-1))/(mu^2 delta)
```

## 5. Consequence

The Engram residual scale `alpha` is not just an implementation detail. It is a
stability control:

```text
larger alpha -> stronger memory injection but larger collision-noise effect
larger K M   -> lower collision-noise effect
larger N     -> higher collision-noise effect
```

This bridges C back to the architecture:

```text
Engram can be safe when the active key set N, hash capacity K M, injection scale
alpha, and downstream margin mu satisfy the inequality above.
```

The next empirical check should log:

```text
||eta||_2
alpha ||eta||_2
downstream logit margin mu
decision flip rate under synthetic collisions
```
