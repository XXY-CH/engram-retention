# Three-Part Original Paper Audit

Created: 2026-05-03

Purpose: re-check the three designed components against the downloaded paper
texts and identify mismatches before continuing proof or implementation.

Audited components:

```text
1. RetNet backbone
2. Engram external/conditional memory
3. Block Attention Residuals, plus optional Sparse Anchor Residual extension
```

Primary local sources:

```text
references/papers/core/retnet_2307.08621.pdf
references/papers/core/engram_conditional_memory_2601.07372.pdf
references/papers/related/attention_is_all_you_need_1706.03762.pdf
references/papers/related/memorizing_transformers_2203.08913.pdf
references/papers/related/deepnet_2203.00555.pdf
references/papers/related/rmsnorm_1910.07467.pdf
references/papers/related/hash_embeddings_1709.03933.pdf
papers/attention-residual/Attention Residuals.pdf
papers/attention-residual/Touvron2021_LayerScale_Residual_Scaling.pdf
papers/attention-residual/He2016_Deep_Residual_Learning.pdf
papers/attention-residual/Srivastava2015_Highway_Networks.pdf
papers/engram/Engram2026_Memory_Consolidation.pdf
```

Extracted text evidence:

```text
tmp/pdf_extracts/retnet_2307.08621.txt
tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt
tmp/pdf_extracts/attention_is_all_you_need_1706.03762.txt
tmp/pdf_extracts/memorizing_transformers_2203.08913.txt
tmp/pdf_extracts/deepnet_2203.00555.txt
tmp/pdf_extracts/rmsnorm_1910.07467.txt
tmp/pdf_extracts/hash_embeddings_1709.03933.txt
tmp/pdf_extracts/local_attention_residuals.clean.txt
tmp/pdf_extracts/local_layerscale_2021.txt
tmp/pdf_extracts/local_resnet_2016.txt
tmp/pdf_extracts/local_highway_networks_2015.txt
tmp/pdf_extracts/local_engram_memory_consolidation.clean.txt
```

Note: `papers/` and `Papers/` are the same directory entries on this filesystem
for the inspected subdirectories, so this audit uses the lowercase `papers/`
paths as canonical.

## 1. RetNet Backbone

### Original-paper support

The RetNet paper directly supports:

- retention as a replacement for multi-head attention;
- parallel, recurrent, and chunkwise recurrent computation forms;
- recurrent representation with `O(1)` inference cost;
- chunkwise recurrent representation for long-sequence modeling with linear
  complexity;
- exponential decay in the retention mask/state update;
- no Transformer-style KV cache requirement for the RetNet backbone.

Evidence:

- `tmp/pdf_extracts/retnet_2307.08621.txt:4-14` says RetNet supports parallel,
  recurrent, and chunkwise recurrent forms, and recurrent representation enables
  low-cost `O(1)` inference.
- `tmp/pdf_extracts/retnet_2307.08621.txt:16-26` gives the decay mask/state
  update structure with `gamma`.
- `tmp/pdf_extracts/retnet_2307.08621.txt:34-39` says recurrent RetNet has
  `O(1)` inference and linear long-sequence memory complexity.
- `tmp/pdf_extracts/retnet_2307.08621.txt:47-50` says recurrent representation
  is efficient in memory/computation and avoids KV-cache tricks.

### Match with current design

Current claim:

```text
RetNet is the streaming main backbone that keeps per-token inference state small.
```

Verdict: supported.

Required precision:

```text
RetNet gives O(1) recurrent inference for the backbone, not O(1) perfect
long-context memory for arbitrary facts or reasoning constraints.
```

For the active Dense baseline this becomes:

```text
O(1) RetNet + O(N_depth) Block AttnRes + O(K) Engram + Dense MLP/FFN
```

The older `O(B)` sparse-anchor term is optional only if token-time exact anchors
are reintroduced.

### Corrections / caveats

1. The proof should use `||Gamma||_op <= gamma < 1`, not only spectral radius,
   unless a non-normality constant is added.
2. RetNet does not eliminate the need for long-range anchors if the task
   requires exact retrieval of distant critical conditions.
3. RetNet paper's GroupNorm/SubLN details mean our normalization assumptions are
   not decorative. A5 is necessary.

## 2. Engram External / Conditional Memory

### Original-paper support

The Engram paper directly supports:

- conditional memory as a complementary sparsity axis to conditional
  computation;
- N-gram lookup with deterministic hashing and multi-head hashing;
- context-aware gating/fusion with hidden states;
- residual integration into the backbone;
- selected-layer placement, not every layer;
- host-memory offload and deterministic prefetch;
- Zipfian / multi-level cache hierarchy motivation;
- reported gains in knowledge, reasoning, code/math, and long-context
  benchmarks;
- embedding parameters trained with Adam in their setup.

Evidence:

- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:5-13`
  supports conditional memory, O(1) lookup, reasoning/code/math gains, and
  prefetching from host memory.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:24-28`
  frames Engram as conditional memory for static embeddings and local lookup.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:42-57`
  defines retrieval/fusion, compressed suffix N-grams, deterministic hashing,
  multi-head hash heads, and multiplicative-XOR hash.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:74-77`
  states residual integration `H^(l) <- H^(l) + Y`.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:141-149`
  supports Zipfian cache hierarchy and placement tradeoff.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:242-260`
  supports training sharding, inference offload to host memory, async prefetch,
  and the 100B offload throughput result.
- `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt:245`
  says embedding parameters are updated with Adam; this is not online RL or
  direct slot editing.

Hash-embedding background:

- `tmp/pdf_extracts/hash_embeddings_1709.03933.txt:78-93` explains bucket
  collisions and multiple hash functions.
- `tmp/pdf_extracts/hash_embeddings_1709.03933.txt:230-243` gives collision
  probability and says multiple independent hash functions reduce total
  collisions.

### Match with current design

Current claim:

```text
Engram handles static/semi-static knowledge lookup, can be offloaded/loaded, and
can support future session-cache experiments.
```

Verdict: mostly supported, with one important boundary.

Supported:

```text
static/semi-static lookup
deterministic hashed N-gram retrieval
multi-head hashing
context-aware gating
residual injection
host-memory offload/prefetch
cache hierarchy motivation
```

Not directly supported:

```text
online RL-only Engram updating
arbitrary non-parametric direct slot edits
session cache correctness and eviction policy
multimodal semantic alignment
```

### Corrections / caveats

1. Our C1/C2/C3 proof should be presented as a safety theorem for an extension,
   not as something the Engram paper already proves.
2. The paper trains embedding parameters with Adam. It does not prove that
   direct slot edits are safe. ROME/MEMIT only support the broader model-editing
   context, not Engram-specific slot edits.
3. Engram uses specific selected layers and multi-branch architecture in the
   paper. Applying it to RetNet is plausible because RetNet has residual hidden
   streams, but placement/gating must be revalidated.
4. The Engram paper says lookup can free attention capacity for global context.
   It does not replace our sparse attention-residual module for explicit CoT
   anchors.

## 3. Block Attention Residuals And Optional Long-Range Sparse Anchors

### Original-paper support

The root `papers/attention-residual/` folder adds an important missing source:

```text
papers/attention-residual/Attention Residuals.pdf
```

This paper directly supports "Attention Residuals" as a depth-wise replacement
for fixed residual accumulation. It does not directly support our earlier
time/token-wise sparse CoT-anchor cache on top of RetNet.

So there are two related but distinct ideas:

```text
Paper AttnRes:
  attention over previous layer/block outputs across depth

Optional Sparse Anchor Residual:
  attention over selected earlier token anchors across time/context
```

There is still no audited paper that directly proposes our exact module:

```text
dynamic optional Sparse Anchor Residual for CoT anchors on top of RetNet
```

But the design is now better supported by several primitives:

1. Attention Residuals replaces fixed residual accumulation with softmax
   attention over preceding layer/block outputs.
2. Transformer attention maps queries to key-value pairs and computes a weighted
   sum of values.
3. Transformer uses residual connections around attention/FFN sublayers.
4. Self-attention can connect positions with constant sequential path length.
5. Memorizing Transformers use external cached key-value pairs to retrieve
   distant exact values.
6. DeepNet and LayerScale study residual scaling/normalization for stable deep
   models.
7. RMSNorm paper supports RMSNorm as a cheaper normalization alternative, but
   does not directly prove our Jacobian lower-bound condition.

Evidence:

- `tmp/pdf_extracts/local_attention_residuals.clean.txt:8-18` says AttnRes
  replaces fixed residual accumulation with softmax attention over preceding
  layer outputs and introduces Block AttnRes to reduce memory footprint.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:91-120` explains
  residuals as gradient highways and depth-wise aggregation, then proposes
  learned softmax attention over depth with block summaries.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:129-136` states
  Block AttnRes reduces memory/communication from `O(Ld)` to `O(Nd)` and has
  low inference overhead.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:321-327` says Block
  AttnRes reduces memory from `O(L)` to `O(N)` and bounds the KV cache by fixed
  block count `N`.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:604-617` reports more
  bounded hidden-state magnitudes and more uniform gradient distribution.
- `tmp/pdf_extracts/attention_is_all_you_need_1706.03762.txt:120-123` states
  residual connection plus LayerNorm around each sublayer.
- `tmp/pdf_extracts/attention_is_all_you_need_1706.03762.txt:133-148` defines
  query/key/value attention and weighted sum over values.
- `tmp/pdf_extracts/attention_is_all_you_need_1706.03762.txt:196-209` describes
  attention using keys/values from encoder or same previous layer, with decoder
  masking for autoregressive use.
- `tmp/pdf_extracts/attention_is_all_you_need_1706.03762.txt:261-272` says
  self-attention gives shorter paths and connects positions with constant
  sequential operations.
- `tmp/pdf_extracts/memorizing_transformers_2203.08913.txt:5-14` supports an
  external non-differentiable memory of recent key-value pairs and exact-value
  distant retrieval.
- `tmp/pdf_extracts/memorizing_transformers_2203.08913.txt:23-26` notes the
  clearability advantage of external memory.
- `tmp/pdf_extracts/deepnet_2203.00555.txt:4-14` supports residual scaling and
  normalization as stability tools.
- `tmp/pdf_extracts/local_layerscale_2021.txt:155-175` supports learnable
  per-channel residual scaling initialized small for deeper Transformers.
- `tmp/pdf_extracts/rmsnorm_1910.07467.txt:14-19` supports RMSNorm as a simpler
  normalization with re-scaling invariance.

### Match with current design

Current claim:

```text
Optional Sparse Anchor Residual preserves selected long-range reasoning anchors that
RetNet's exponential decay may otherwise weaken.
```

Verdict: plausible and consistent with primitives, but not directly claimed by
the audited papers as a token-time RetNet CoT-anchor module.

Supported primitives:

```text
depth-wise attention residual aggregation
attention over key-value pairs
residual connections
short path length for attention
external key-value memory
residual/normalization stability concerns
```

Optional extension:

```text
dynamic sparse gate chooses CoT anchors
cache budget B
gradient non-decay theorem under selected/read/dominant events
RL or adaptive memory-price training for the gate
```

This extension is not directly in the original papers and must remain our own
conditional theorem plus experiment target.

### Corrections / caveats

1. Do not cite Attention Residuals/Transformer/DeepNet/Memorizing Transformers
   as proving our dynamic CoT anchor gate. They provide strong primitives, but
   AttnRes is depth-wise, while our anchor cache is token/time-wise.
2. A1/A2/A3/A4/A5 are therefore necessary original proof work, not just a
   restatement of literature.
3. To avoid ambiguity, use two names:

```text
Depth-wise Attention Residuals / AttnRes:
  the Kimi paper's layer/block-output aggregation mechanism.

Sparse Anchor Residual:
  our token-time side path that stores selected KV anchors and later reads them
  by attention.
```

If we still use "attention residual" informally, define it as:

```text
a sparse residual side path that stores selected key-value anchors and later
reads them by attention
```

not as if it were a standard named module from the papers.

## 4. Net Drift Assessment

| Component | Current design vs original papers | Drift level | Required correction |
|---|---:|---:|---|
| RetNet backbone | Strongly aligned | Low | Keep `O(1)` scoped to backbone recurrent inference. |
| Engram memory | Mostly aligned | Medium | Keep online/session updates as conditional extensions. |
| Block Attention Residuals | Directly supported as depth-wise AttnRes | Low | Use Kimi AttnRes only for depth/layer/block-output aggregation. |
| Sparse Anchor Residual | Token-time anchor module is ours, optional, and not in the active Dense baseline | High | Cite it as project extension, not as Kimi AttnRes. |

## 5. Impact On Existing Proof Chain

No core proof needs to be discarded, but the active/appendix boundary changes.

Required wording changes:

```text
RetNet theorem = paper-backed backbone assumption.
Engram retrieval/offload theorem = paper-backed with our hash-noise safety
extension.
Block AttnRes theorem = active depth-axis proof target, directly motivated by
Kimi AttnRes.
Sparse Anchor Residual theorem = optional appendix based on attention/KV-memory
and residual-stability primitives, not a direct literature result.
```

The old closure condition remains valid only for the optional Sparse Anchor
Residual:

```text
r q d > |Y|delta_C + delta_B
```

It should not be used as the main closure condition for the active Dense RetNet
+ Block AttnRes + Engram baseline.

## 6. Next Correction To Carry Forward

In future writing, use the following three labels:

```text
RetNet backbone: literature-supported component.
Engram conditional memory: literature-supported component with extension-safe
proofs for session/update behavior.
Depth-wise AttnRes: literature-supported residual aggregation primitive.
Sparse Anchor Residual: optional novel token-time component, supported by
attention/KV-memory primitives and our conditional proof.
```

This prevents the project from accidentally overstating paper support.
