# PDF Assumption Audit: A1 And C1

Created: 2026-05-03

Purpose: check the proof assumptions in `02-conditional-nondecay-gradient-proof.md`
and `03-engram-hash-capacity-proof.md` against the downloaded PDFs.

## Sources Inspected

Local PDFs:

- `references/papers/core/retnet_2307.08621.pdf`
- `references/papers/related/deepnet_2203.00555.pdf`
- `references/papers/related/feature_hashing_0902.2206.pdf`
- `references/papers/related/count_min_sketch_2005.pdf`
- `references/papers/core/vq_vae_1711.00937.pdf`
- `references/papers/related/memorizing_transformers_2203.08913.pdf`

Extraction notes are in `tmp/pdf_extracts/`.

## Verdict

The proof direction is broadly consistent with the papers, but two boundaries
must be tightened:

1. A1 is valid for the target RetNet-style recurrence only under a contractive
   decay operator. The RetNet paper uses fixed per-head decay values below one,
   but the current repository implementation uses a trainable `gamma.abs()`,
   which can exceed one unless constrained.
2. C1's signed-noise proof is a feature-hashing-style adaptation. Count-Min
   Sketch supports the multi-hash/pairwise-independence theme, but it is not the
   same signed vector estimator and should not be cited as proving zero-mean
   collision noise.

## A1 Check: RetNet Decay And Residual Path

### What the PDF supports

RetNet defines a recurrent retention form:

```text
S_n = gamma S_{n-1} + K_n^T V_n
Retention(X_n) = Q_n S_n
```

and defines parallel retention with a causal exponential decay mask:

```text
D_nm = gamma^(n-m), n >= m
D_nm = 0, n < m
```

The paper also states that recurrent representation enables low-cost `O(1)`
inference and that chunkwise recurrence supports efficient long-sequence
modeling with linear complexity.

The paper assigns different fixed per-head decay values below one. The extracted
text includes:

```text
gamma = 1 - 2^(-5 - arange(0,h))
```

and another experimental setting:

```text
gamma = 1 - exp(linspace(log 1/32, log 1/512, h))
```

So the simplified proof assumption:

```text
||Gamma||_op <= gamma < 1
```

is faithful to the intended RetNet decay mechanism if `Gamma` is scalar or
diagonal per head and all head decays are bounded below one.

### What needed correction

The proof must not claim to apply directly to the current code without changing
the decay parameterization:

```python
self.gamma = nn.Parameter(torch.ones(self.n_heads) * 0.99)
...
gamma = self.gamma.abs()
```

`abs()` ensures nonnegative decay, but it does not ensure decay below one.
Therefore a trained model could violate the contractive assumption. To align the
implementation with the proof, use one of:

```text
fixed RetNet-style gamma schedule
gamma = sigmoid(raw_gamma)
gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(raw_gamma), gamma_max < 1
```

Follow-up applied: `src/layers/retention.py` now uses a fixed RetNet-style
per-head gamma buffer below one, and `tests/test_retention.py` checks the
contractive invariant.

### Residual path conclusion

No PDF contradicts the residual-cache value-path proof. It is an added target
architecture, not a RetNet claim. The proof correctly treats it as a conditional
path:

```text
a_i = 1
p_{T,i} >= epsilon
alignment c_align > 0
```

The proof should continue to avoid claiming that sparse residual attention
guarantees long-context reasoning by itself.

### Normalization caveat

DeepNet supports the general idea that residual scaling/normalization affects
stability and can bound updates, but it does not prove our specific
LayerNorm/RMSNorm lower-bound assumption. The A1 proof therefore correctly uses
a separate directional alignment assumption instead of citing DeepNet as a direct
proof of non-degenerate `J_N`.

## C1 Check: Engram Hash Capacity

### What Feature Hashing supports

The Feature Hashing paper explicitly defines:

```text
h: N -> {1, ..., m}
xi: N -> {+/- 1}
```

and a signed hashed feature map. It states that the signed modification gives an
unbiased estimate, and its Lemma 2 gives an unbiased hash kernel with variance
`O(1/m)` for unit vectors.

This supports C1's use of signed hashes to make collision noise zero-mean and
to expect variance shrinking with more slots.

### What Count-Min supports and does not support

The Count-Min Sketch paper supports:

- multi-row hash sketches;
- pairwise independent hash functions;
- probabilistic error bounds controlled by sketch width/depth.

But Count-Min uses nonnegative counters and a `min` estimator for point queries.
It does not prove the zero-mean signed vector-noise estimator used in C1.

Therefore C1 should cite Count-Min only for the multi-hash sketch design pattern,
not for signed-noise unbiasedness.

### What needed correction

The all-head collision bound:

```text
(N-1)/M^K
```

controls the event that the same wrong key collides in every head. It does not
control all retrieval error, because one-head collisions still contribute noise
to the averaged vector. The proof already has a separate variance bound:

```text
E[||eta_x||^2] <= ((N-1) R^2)/(K M)
```

This distinction should remain explicit.

### Multimodal check

VQ-VAE supports the premise that continuous inputs can be mapped to discrete
latent codes. It does not prove semantic alignment. The proof correctly treats
namespace/salt as collision hygiene, not as a semantic-binding solution.

## Required Follow-Up Edits

- Keep A1 marked as a target-design proof for sparse residual anchors, while the
  RetNet backbone decay invariant is now aligned with the PDF.
- Keep the implementation note in `02-conditional-nondecay-gradient-proof.md`.
- Add a citation caveat to `03-engram-hash-capacity-proof.md` separating Feature
  Hashing from Count-Min.
- Later, add a small code-level test that fails if any retention head has
  `gamma >= 1`.
