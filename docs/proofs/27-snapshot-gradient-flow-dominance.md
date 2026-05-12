# Conditional Note S1: Snapshot-to-Logit Gradient Advantage

Created: 2026-05-04

Status: conditional optimization note. It identifies a plausible gradient
advantage for exact-copy tasks but does not guarantee automatic module
activation.

## 0. Goal

A critical engineering vulnerability identified earlier is "Lazy Learning": if the Snapshot readout is initialized with a tiny weight or bias, the model might optimize the dense backbone (FFN) instead, leaving the Snapshot module "dead". 

The useful goal is narrower: identify conditions under which the
Snapshot-to-Logit path receives a stronger direct gradient than a dense path
that must reconstruct a high-entropy token from a saturated recurrent state.
This motivates branch-aware optimization; it does not remove the need for
curriculum, learning-rate control, or causal ablations.

## 1. Network Path Definition

Let the final output logit for token $v$ at time $T$ be $y_{T, v}$. It receives additive contributions from two paths:
$$ y_{T, v} = y_{T, v}^{Dense} + \lambda_{snap} y_{T, v}^{Snap} $$

Where:
1. **Dense Path**: $y_{T}^{Dense} = W_{head} \mathcal{F}_L( \dots \mathcal{F}_1(S_T) \dots )$, heavily dependent on the saturated recurrent state $S_T$.
2. **Snapshot Path**: $y_{T}^{Snap} = \text{Softmax}(Q_T K_i^T) V_i$. Note that $V_i$ represents the exact token embedding of the needle.

## 2. Gradient Flow Attenuation

Consider the loss $\mathcal{L}_T = -\log \text{Softmax}(y_T)_{v^*}$, where $v^*$ is the correct target token.
The gradient of the loss w.r.t the parameters of the two paths dictates the optimization speed.

**Dense Path Gradient:**
$$ \left\| \frac{\partial \mathcal{L}_T}{\partial W_1} \right\| = \left\| \frac{\partial \mathcal{L}_T}{\partial y_T} \frac{\partial y_T}{\partial h_L} \left( \prod_{l=1}^{L-1} J_{\mathcal{F}_l} \right) \frac{\partial \mathcal{F}_1}{\partial W_1} \right\| \le C_D \rho^{L} $$
where $J_{\mathcal{F}_l}$ are the Jacobians of the dense layers. For high-entropy tasks where the correlation between $S_T$ and $v^*$ is destroyed by superposition, the expected dot product $\langle \frac{\partial \mathcal{L}_T}{\partial y_T}, \text{Dense\_Output} \rangle$ is near zero, causing gradient starvation.

**Snapshot Path Gradient:**
$$ \left\| \frac{\partial \mathcal{L}_T}{\partial \lambda_{snap}} \right\| = \left| \frac{\partial \mathcal{L}_T}{\partial y_{T, v^*}} \cdot y_{T, v^*}^{Snap} \right| $$
Because $y_{T}^{Snap}$ maps *directly* to the token embedding $V_i$ via the attention score, if $V_i$ is the correct answer $v^*$, the term $y_{T, v^*}^{Snap}$ is large ($\mathcal{O}(1)$).

## 3. Conditional Gradient-Advantage Statement

Let $t$ be the training step in continuous time (Gradient Flow). The dynamics of the parameters follow $\dot{\theta} = -\eta \nabla_\theta \mathcal{L}$.

If the Dense path gradient is attenuated by both depth $L$ and state saturation
noise, while the snapshot readout has captured the correct value and produces a
positive logit-margin direction, then the snapshot branch can have a larger
direct gradient:
$$ \left\| \nabla_{\lambda_{snap}} \mathcal{L} \right\| \gg \max_{l} \left\| \nabla_{W_l} \mathcal{L} \right\| $$

**Conclusion:**
For high-entropy exact-copy tasks, a captured snapshot-to-logit path supplies a
short, interpretable gradient route. This is a conditional inductive bias, not a
guarantee that the module will awaken automatically. The practical contract is
to report branch scales, snapshot attention mass, module-drop deltas, and
training settings that could suppress small residual branches.
