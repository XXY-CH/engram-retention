# Conditional Note M1: MoE Routing Perturbation under Residual Injection

Created: 2026-05-04

Status: Phase-2 risk note. Establishes a local perturbation bound for router
probabilities; it does not prove global MoE training stability.

## 0. Goal

The third-party critique noted that deferring MoE sacrifices parameter scaling.
We established MoE deferral as a Phase-1 scaffold to isolate signals. For Phase
2, the first mathematical obligation is narrower than "prove no collapse": bound
how much small residual injections can perturb a router at initialization and
identify the additional training-time checks needed before any scaling claim.

## 1. Lipschitz Continuity of the MoE Router

An MoE router for $E$ experts computes the assignment probabilities $p(x) \in \mathbb{R}^E$ for an input vector $x \in \mathbb{R}^d$:
$$ p_i(x) = \frac{\exp(w_i^T x / \tau)}{\sum_{j=1}^E \exp(w_j^T x / \tau)} $$
where $w_i$ are the routing weights and $\tau$ is the temperature.

Let the Jacobian of the router be $J_p(x) \in \mathbb{R}^{E \times d}$. The partial derivatives of the softmax function are well-known:
$$ \frac{\partial p_i}{\partial x} = \frac{1}{\tau} p_i(x) \left( w_i - \sum_{j=1}^E p_j(x) w_j \right)^T $$
From this, the spectral norm of the Jacobian is strictly bounded by the norms of the routing weights:
$$ \|J_p(x)\|_{op} \le \frac{1}{\tau} \max_i \|w_i\|_2 $$
Thus, the routing function $p(x)$ is globally Lipschitz continuous with constant $L_{route} = \frac{1}{\tau} \max_i \|w_i\|_2$.

## 2. Bounding the Routing Perturbation

Let the injected residual from Engram and Block AttnRes be $\Delta x = \lambda_E \mathcal{E}(x) + \lambda_A \mathcal{A}(x)$. 
From our Composition Guard (Proof 19), we initialize $|\lambda_E|, |\lambda_A| \le \epsilon_{scale}$ such that $\|\Delta x\|_2 \le \tau_{inject}$.

By the Lipschitz property, the shift in routing probabilities caused by our architecture is bounded by:
$$ \| p(x + \Delta x) - p(x) \|_1 \le \sqrt{E} \| p(x + \Delta x) - p(x) \|_2 \le \sqrt{E} L_{route} \|\Delta x\|_2 \le \sqrt{E} L_{route} \tau_{inject} $$

## 3. Local Router-Perturbation Bound

Let $\mathcal{B}(x) = -\sum_{i=1}^E p_i(x) \log p_i(x)$ be the entropy of the routing distribution, which serves as a proxy for load balancing (higher entropy = more balanced routing). 

Since entropy is locally Lipschitz continuous for probability distributions bounded away from 0, the change in routing entropy is bounded:
$$ |\mathcal{B}(x + \Delta x) - \mathcal{B}(x)| \le C_{entropy} \sqrt{E} L_{route} \tau_{inject} $$

**Conclusion:**
With small LayerScale initialization ($\epsilon_{scale} = 10^{-4}$) and bounded
branch outputs, the initial routing distribution is close to the unperturbed
baseline routing distribution by the Lipschitz bound above.

This does not guarantee that load-balancing dynamics are perfectly preserved,
nor that the router cannot collapse during training. Training-time stability
also depends on router temperature, auxiliary load-balancing loss, expert
capacity, token dropping, gradient clipping, optimizer state, and the growth
schedule of the injected branches. Phase-2 validation must therefore log router
entropy, per-expert load, dropped-token rate, branch norms, and loss divergence
under MoE-specific ablations.
