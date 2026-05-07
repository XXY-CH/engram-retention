# Tight Engram Concentration Bound

Created: 2026-05-04

Status: refinement of the earlier Engram collision-noise proof. This replaces a
loose Markov-style tail interpretation with a sharper high-probability bound
under stronger assumptions.

## 0. Claim Boundary

The bound applies to a signed, zero-mean, multi-head hash estimator with bounded
or sub-Gaussian stored vectors. It is an idealized proof target. The current
implementation must be checked against the signed/zero-mean assumptions before
using this theorem as evidence.

## 1. Noise Model

For query key `x`, collision noise is:

```text
eta_x =
(1/K) sum_{k=1}^K sum_{x' != x}
I_{k,x'} sigma_{k,x'} e_{x'}
```

where:

```text
I_{k,x'} = 1[h_k(x') = h_k(x)]
Pr(I_{k,x'} = 1) = 1/M
sigma_{k,x'} in {-1, +1}
||e_{x'}||_2 <= R
```

Assume independence across heads and Rademacher signs.

## 2. Directional Sub-Gaussian Bound

For any fixed unit vector `u`, define:

```text
X_{k,x'} = (1/K) I_{k,x'} sigma_{k,x'} <u, e_{x'}>
```

Then:

```text
|X_{k,x'}| <= R/K
E[X_{k,x'}] = 0
sum Var(X_{k,x'}) <= (N - 1) R^2 / (K M)
```

By Bernstein's inequality:

```text
Pr(|<u, eta_x>| >= t)
<= 2 exp(
  - t^2 / (2 v + (2 R t)/(3K))
)
```

where:

```text
v = (N - 1) R^2 / (K M)
```

For moderate `t`, the directional noise is sub-Gaussian at scale:

```text
sqrt((N R^2)/(K M))
```

## 3. Vector Norm Bound By Net Argument

Let `S^{d-1}` be covered by a `1/2`-net with size at most `5^d`. If:

```text
max_{u in net} |<u, eta_x>| <= a
```

then:

```text
||eta_x||_2 <= 2a
```

Thus, with probability at least `1 - delta`:

```text
||eta_x||_2
<= 2 sqrt(
  2 v log(2 * 5^d / delta)
)
```

up to the Bernstein linear correction term.

Substituting `v`:

```text
||eta_x||_2
= O(
  R sqrt( (N (d + log(1/delta))) / (K M) )
)
```

## 4. Margin-Safe Retrieval Condition

Let the downstream classifier or residual decision have margin `m`. If Engram is
injected with scale `lambda_E`, a sufficient high-probability safety condition is:

```text
lambda_E ||eta_x||_2 <= m / L_down
```

where `L_down` bounds downstream sensitivity.

Using the concentration bound, it is sufficient that:

```text
lambda_E R sqrt( (N (d + log(1/delta))) / (K M) )
<= c m / L_down
```

for a universal constant `c`.

## 5. Why This Is Tighter Than Markov

The older variance-plus-Markov route gives:

```text
Pr(||eta_x||^2 >= a) <= E||eta_x||^2 / a
```

which decays only polynomially. The signed concentration route gives exponential
tails, but only under stronger assumptions:

```text
independent signs
bounded/sub-Gaussian vectors
stable downstream Lipschitz margin
```

## 6. Experimental Obligations

To use this theorem, report:

```text
N active keys
K hash heads
M slots
d
estimated collision norm distribution
lambda_E
module-drop Engram delta
```

Without these, the theorem remains a design guide rather than evidence.
