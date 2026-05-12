# Conditional Theorem: TokenCopyBuffer Provides an Exact-Copy Logit Path

Created: 2026-05-10

Status: conditional mechanism theorem, supported by synthetic needle diagnostics.

## 0. Context

In the current synthetic needle diagnostics, the RetNet-only variants do not
recover the marked high-entropy token reliably. The recurrent state compresses
token-level information below recoverable fidelity for this task. The
TokenCopyBuffer adds a separate value path by storing raw token embeddings in a
bounded external buffer and projecting the readout through the embedding weight
matrix.

## 1. Setup

Let $E \in \mathbb{R}^{V \times d}$ be the token embedding matrix.
For input token $t_i$ at position $i$, the stored embedding is $e_i = E[t_i]$.

The TokenCopyBuffer maintains a set of stored embeddings $\{e_{s_1}, ..., e_{s_K}\}$
at marked source positions $\{s_1, ..., s_K\}$ where $K \leq$ `max_snapshots`.

At output position $j$, the copy readout is:

$$r_j = \sum_{k=1}^{K} \alpha_{jk} \cdot e_{s_k}$$

where $\alpha_{jk}$ are attention weights ($\alpha_{jk} \geq 0$, $\sum_k \alpha_{jk} = 1$).

The copy logits added to the base model are:

$$c_j = r_j \cdot E^T \in \mathbb{R}^V$$

## 2. Claim

If all of the following conditions hold:

1. The target token $t^*$ is captured in the bounded buffer.
2. The copy attention concentrates on the captured target slot, i.e.
   $\alpha_{jk^*}=1$ or more generally $\alpha_{jk^*}$ is sufficiently close to
   one.
3. The embedding matrix has a strict self-similarity margin for $t^*$:

$$
\|E[t^*]\|^2 > \max_{v \ne t^*} E[t^*]\cdot E[v].
$$

4. The resulting copy-logit margin exceeds any wrong-token advantage created by
   the base logits.

Then the TokenCopyBuffer provides an exact-copy logit path for $t^*$ at output
position $j$.

Under perfect attention concentration, the copy logit for $t^*$ equals the
squared norm of its embedding:

$$c_j[t^*] = \|e_{t^*}\|^2$$

This is not, by itself, a guarantee of exact prediction; it becomes one only
when the vocabulary margin and base-logit margin conditions above hold.

## 3. Proof

With $\alpha_{jk} = 1$ for $k = k^*$ (the stored position containing $t^*$):

$$r_j = e_{s_{k^*}} = E[t^*]$$

$$c_j[v] = e_{s_{k^*}} \cdot E[v] = E[t^*] \cdot E[v]$$

For $v = t^*$:

$$c_j[t^*] = E[t^*] \cdot E[t^*] = \|E[t^*]\|^2$$

For any competitor token $v$:

$$c_j[v] = E[t^*] \cdot E[v] \leq \|E[t^*]\| \cdot \|E[v]\|$$

The strict condition needed for copy-logit ranking is:

$$
\Delta_{\mathrm{copy}}
= c_j[t^*] - \max_{v\ne t^*} c_j[v] > 0.
$$

For the final model logits $z_j = z_j^{\mathrm{base}} + c_j$, exact prediction is
guaranteed if:

$$
\Delta_{\mathrm{copy}}
>
\max_{v\ne t^*} z_j^{\mathrm{base}}[v] - z_j^{\mathrm{base}}[t^*].
$$

Near-orthogonal embeddings make this margin plausible at initialization and
trainable in the synthetic setting, but orthogonality is an assumption or an
empirical property to measure, not a universal fact.

QED.

## 4. Scaling Limitation

The proof above assumes perfect or sufficient attention concentration.
In practice, the attention weights are computed from hidden states:

$$\alpha_{jk} = \text{softmax}_k \left( \frac{q_j^T \cdot W_k \cdot e_{s_k}}{\sqrt{d}} \right)$$

where $q_j = \text{RMSNorm}(h_j)$ is derived from the final hidden state.
At longer sequences, the hidden states $h_j$ at answer positions must
discriminate among stored tokens. This requires:

1. Positional differentiation: $h_{j_1} \neq h_{j_2}$ for different answer
   positions $j_1, j_2$ (so different stored tokens are selected).
2. Sufficient signal: the query $q_j$ must retain enough information to
   match the correct stored key.

Empirically, content-only keys degrade at seq_len > 256 in the local needle
diagnostic. A likely cause is that hidden states at answer positions become less
differentiated after processing more filler tokens, but this should be treated
as an empirical diagnosis rather than a theorem.

### 4.1 Positional Encoding Resolution

The observed scaling limitation is reduced in the current synthetic diagnostic
by adding a learnable position embedding to the copy attention keys:

$$k_m = W_k \cdot e_{s_m} + P[s_m]$$

where $P[s_m]$ is a position embedding for the source position $s_m$.
This provides the attention mechanism with explicit positional information. It
can discriminate stored tokens by their original position when the query state
also encodes enough information about which source position is needed.

With positional keys, the copy attention can learn a simple positional
alignment: answer position $j$ attends to the stored token whose source position
$s_m$ corresponds to the correct answer offset. This converts the discrimination
problem into a learned positional lookup only when the training distribution
contains a stable offset/alignment rule.

## 5. Empirical Verification

### Without positional encoding (content-only keys)

| seq_len | steps | eval_em |
|---------|-------|---------|
| 128 | 2000 | 0.984 |
| 256 | 4000 | 0.938 |
| 512 | 4000 | 0.797 |

### With positional encoding (content + position keys)

| seq_len | steps | eval_em |
|---------|-------|---------|
| 512 | 400 | **1.000** |
| 1024 | 400 | **1.000** |

Without TokenCopyBuffer, eval_em = 0.000-0.016 in the compared local runs.
These numbers support the conditional mechanism claim on the synthetic needle
task; they do not establish general long-context reasoning or open-domain
retrieval.
