# Conditional Bound C4: Concentration of Signed Engram Collision Noise

Created: 2026-05-04

Status: conditional concentration note. It strengthens the Markov-style sanity
check under idealized signed-hash assumptions, but it is not a guarantee of
deep-network stability.

## 0. Goal

The previous proofs bounded the expected squared norm of the collision noise $\eta_x$ and applied Markov's inequality to bound the tail probability. Markov yields a slow polynomial decay ($\mathcal{O}(1/\delta)$), which is insufficient to guarantee stability for billion-scale parameter training where rare extreme collisions can cause divergence.

Here, we use scalar projections of signed collision noise to obtain an
exponential tail bound under independence, boundedness, and uniform-hashing
assumptions. The result is useful for margin accounting; it does not cover
semantic collisions, trained-table correlations, or downstream optimizer
dynamics by itself.

## 1. Problem Formulation

Recall the retrieved noise for target key $x$:
$$ \eta_x = \frac{1}{K} \sum_{k=1}^K \sum_{x' \neq x} \mathbf{1}[h_k(x') = h_k(x)] s_k(x) s_k(x') e_{x'} $$
where $s_k \in \{-1, +1\}$ are independent Rademacher random variables, $K$ is the number of hash heads, and $h_k(x) \in \{1, \dots, M\}$ is the uniform hash function. $N$ is the total active keys. Assume $\|e_{x'}\|_2 \le R$.

For a fixed hash projection $h_k$, let the indicator $I_{k, x'} = \mathbf{1}[h_k(x') = h_k(x)]$. Note that $\mathbb{E}[I_{k, x'}] = \frac{1}{M}$.

## 2. Hoeffding's Concentration Inequality

Let $Z_k = \sum_{x' \neq x} I_{k, x'} s_k(x) s_k(x') e_{x'}$ be the noise vector from head $k$. 
Conditioned on the hash mappings (which fix $I_{k, x'}$), $Z_k$ is a sum of independent zero-mean bounded random vectors (scaled by Rademacher signs). 

By the vector Hoeffding inequality (or sub-Gaussian concentration), the projection of $\eta_x = \frac{1}{K}\sum_{k=1}^K Z_k$ onto any unit vector $u$ satisfies:
$$ \mathbb{P}(|\langle \eta_x, u \rangle| \ge t) \le 2 \exp\left(-\frac{K^2 t^2}{2 \sum_{k=1}^K \nu_k^2}\right) $$
where $\nu_k^2 = \sum_{x' \neq x} I_{k, x'} R^2$.

## 3. High-Probability Collision-Count Bound

Taking the expectation over the uniform hash choices $I_{k, x'}$, the expected number of collisions per head is $\frac{N-1}{M}$. By applying a Chernoff bound on the binomial sum of collisions, one obtains a high-probability bound of the form:

$$
\sum_{k=1}^{K}\nu_k^2
\le
C_{\mathrm{coll}}K\frac{N+\log(1/\delta_{\mathrm{coll}})}{M}R^2
$$

for an absolute constant $C_{\mathrm{coll}}$, up to the usual binomial-tail
constant factors. This is not an unconditional deterministic bound; the
collision-count failure probability must be added to the final tail bound.

Substituting this back into the sub-Gaussian tail bound:
$$
\mathbb{P}(|\langle\eta_x,u\rangle| \ge \sigma)
\le
2 \exp\left(
- \frac{C \cdot K \cdot M \cdot \sigma^2}
{(N+\log(1/\delta_{\mathrm{coll}}))R^2}
\right)
+ \delta_{\mathrm{coll}}
$$
for any fixed unit vector $u$. A norm bound requires a covering argument or a
dimension-dependent vector concentration inequality, so it should include the
corresponding dimension factor.

## 4. Conditional Margin Bound

If a downstream margin calculation only depends on the projection of the noise
onto a fixed unit direction $u$, and if it can tolerate projected perturbation
at most $\mu/\alpha$, then:

$$
\mathbb{P}(|\alpha\langle\eta_x,u\rangle| \ge \mu)
\le
2 \exp\left(
- \frac{C K M \mu^2}
{\alpha^2 (N+\log(1/\delta_{\mathrm{coll}}))R^2}
\right)
+ \delta_{\mathrm{coll}}.
$$

**Conclusion:** 
Under the idealized signed-hash assumptions, projected collision noise has an
exponential tail in the effective hash budget $KM/N$. This is stronger than the
earlier Markov sanity check, but it is not an absolute guarantee that Engram
cannot perturb a deep network. Practical safety still requires bounded branch
scale, normalization, collision diagnostics, causal-drop evaluation, and
downstream margin measurements.
