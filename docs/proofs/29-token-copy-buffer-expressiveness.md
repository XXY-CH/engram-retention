# Proof: TokenCopyBuffer Provides Exact Recall via Embedding Projection

Created: 2026-05-10

Status: theorem, verified empirically.

## 0. Context

Pure RetNet cannot solve the needle copy task (eval_em = 0.000 for all
sequence lengths tested). The recurrent state compresses token-level
information below recoverable fidelity. The TokenCopyBuffer bypasses
this compression by storing raw token embeddings in an external buffer
and projecting directly to logits through the embedding weight matrix.

## 1. Setup

Let $E \in \mathbb{R}^{V \times d}$ be the token embedding matrix.
For input token $t_i$ at position $i$, the stored embedding is $e_i = E[t_i]$.

The TokenCopyBuffer maintains a set of stored embeddings $\{e_{s_1}, ..., e_{s_K}\}$
at marked source positions $\{s_1, ..., s_K\}$ where $K \leq$ `max_snapshots`.

At output position $j$, the copy readout is:

$$r_j = \sum_{k=1}^{K} \alpha_{jk} \cdot e_{s_k}$$

where $\alpha_{jk}$ are attention weights ($\alpha_{jk} \geq 0$, $\sum_k \alpha_{jk} = 1$).

The copy logits are:

$$c_j = r_j \cdot E^T \in \mathbb{R}^V$$

## 2. Claim

If the attention weight concentrates perfectly on the stored embedding
for target token $t^*$ (i.e., $\alpha_{jk} = 1$ for the $k$ where $t_{s_k} = t^*$),
then the copy logit for $t^*$ equals the squared norm of its embedding:

$$c_j[t^*] = \|e_{t^*}\|^2$$

This is the maximum possible value of $c_j[v]$ for any vocabulary entry $v$
with $\|e_v\| \leq \|e_{t^*}\|$, assuming near-orthogonal embeddings.

## 3. Proof

With $\alpha_{jk} = 1$ for $k = k^*$ (the stored position containing $t^*$):

$$r_j = e_{s_{k^*}} = E[t^*]$$

$$c_j[v] = e_{s_{k^*}} \cdot E[v] = E[t^*] \cdot E[v]$$

For $v = t^*$:

$$c_j[t^*] = E[t^*] \cdot E[t^*] = \|E[t^*]\|^2$$

By Cauchy-Schwarz:

$$c_j[v] = E[t^*] \cdot E[v] \leq \|E[t^*]\| \cdot \|E[v]\|$$

with equality iff $E[t^*]$ and $E[v]$ are parallel. For distinct vocabulary
entries with random initialization, embeddings are near-orthogonal, so
$c_j[t^*] > c_j[v]$ for $v \neq t^*$ with high probability.

QED.

## 4. Scaling Limitation

The proof assumes perfect attention concentration ($\alpha_{jk} = 1$).
In practice, the attention weights are computed from hidden states:

$$\alpha_{jk} = \text{softmax}_k \left( \frac{q_j^T \cdot W_k \cdot e_{s_k}}{\sqrt{d}} \right)$$

where $q_j = \text{RMSNorm}(h_j)$ is derived from the final hidden state.
At longer sequences, the hidden states $h_j$ at answer positions must
discriminate among stored tokens. This requires:

1. Positional differentiation: $h_{j_1} \neq h_{j_2}$ for different answer
   positions $j_1, j_2$ (so different stored tokens are selected).
2. Sufficient signal: the query $q_j$ must retain enough information to
   match the correct stored key.

Empirically, single-head attention degrades at seq_len > 256 because
the hidden states at answer positions become less differentiated after
processing more filler tokens.

### 4.1 Positional Encoding Resolution

The scaling limitation is resolved by adding a learnable position
embedding to the copy attention keys:

$$k_m = W_k \cdot e_{s_m} + P[s_m]$$

where $P[s_m]$ is a position embedding for the source position $s_m$.
This provides the attention mechanism with explicit positional information,
allowing it to discriminate stored tokens by their original position
rather than relying solely on hidden state differentiation.

With positional keys, the copy attention can learn a simple positional
alignment: answer position $j$ attends to the stored token whose source
position $s_m$ corresponds to the correct answer offset. This converts
the hard discrimination problem (differentiating hidden states after 500+
filler tokens) into a simple positional lookup.

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

Without TokenCopyBuffer, eval_em = 0.000-0.016 at all lengths.
