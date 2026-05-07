# Gemini Full Alignment And Recommended Papers

Created: 2026-05-03

Purpose: answer whether the rest of the Gemini discussion is consistent with the
current Dense RetNet + Block AttnRes + Engram plan, and list the other
papers worth reading from the Engram paper's citation graph.

## 1. Consistency Check

| Gemini topic | Consistent with current plan? | Notes |
|---|---:|---|
| Engram as knowledge / static-pattern lookup | Yes | Matches Engram PDF: N-gram lookup, deterministic hashing, context-aware gate, residual injection. Fits our C1/C2 proof chain. |
| Engram to improve reasoning by freeing early layers | Yes, as motivation | Engram PDF reports BBH/HumanEval/MATH gains and LogitLens/CKA effective-depth evidence. For RetNet this remains a hypothesis to verify, not a transferred guarantee. |
| Temporary online/session cache to avoid repeated long-context upload | Partially | Deterministic IDs, host offload, prefetch, and multi-level cache hierarchy support the engineering direction. But the PDF does not evaluate user/session cache correctness or eviction. |
| RetNet adaptation | Mostly | Engram is described as topology-agnostic and residual-stream integrated. RetNet has hidden-state layers and residual structure, so adaptation is plausible. But layer placement and normalization must be revalidated. |
| RL updating Engram to keep knowledge current | Not yet | The Engram PDF updates embedding parameters with Adam during pretraining. It does not prove RL-only Engram updates, online updates, or safe non-parametric slot editing. This should remain future work. |
| Direct hash-slot edits for knowledge injection | Speculative | ROME/MEMIT support the broader idea that model memories can be edited, but Engram-specific slot editing needs its own collision/interference/gating proof. |
| Attention residual as continuous thought support | Needs corrected wording | Kimi AttnRes supports depth-wise residual aggregation, not token-time CoT-anchor caching. The old A-time proof chain remains optional appendix material. |

## 2. Current Refined Model Statement

```text
RetNet = streaming backbone for small-memory sequential compute.
Block AttnRes = depth-wise residual aggregation for deep Dense blocks.
Engram = conditional memory for static/semi-static lookup, offload, prefetch,
         and future session-cache experiments.
Sparse Anchor Residual = optional future token-time anchor extension, not active
                         baseline.
```

This is consistent with the Gemini discussion if we keep the RL and session-cache
parts as future hypotheses.

## 3. Recommended Papers From This Discussion

### Must Read For Engram Core

| Priority | Paper | Local file | Why it matters |
|---|---|---|---|
| P0 | Cheng et al., "Conditional Memory via Scalable Lookup" | `references/papers/core/engram_conditional_memory_2601.07372.pdf` | Primary Engram paper. |
| P0 | Svenstrup et al., "Hash Embeddings for Efficient Word Representations" | `references/papers/related/hash_embeddings_1709.03933.pdf` | Direct ancestor for hash embedding tables. |
| P0 | Weinberger et al., "Feature Hashing..." | `references/papers/related/feature_hashing_0902.2206.pdf` | Signed hashing, unbiased collision-noise framing. |
| P0 | Lample et al., "Large Memory Layers with Product Keys" | `references/papers/related/product_key_memory_1907.05242.pdf` | Large neural memory layers and fast lookup. |
| P0 | Liu et al., "Infini-gram..." | `references/papers/related/infini_gram_2401.17377.pdf` | Modern large-scale N-gram lookup system. |

### Must Read For Knowledge Updating Claims

| Priority | Paper | Local file | Why it matters |
|---|---|---|---|
| P0 | Meng et al., "Locating and Editing Factual Associations in GPT" | `references/papers/related/rome_2202.05262.pdf` | Single-fact model editing; useful contrast to Engram slot updates. |
| P0 | Meng et al., "Mass-Editing Memory in a Transformer" | `references/papers/related/memit_2210.07229.pdf` | Many-fact editing; relevant to online knowledge update risk. |
| P1 | Mallen et al., "When Not to Trust Language Models..." | `references/papers/related/nonparametric_memory_trust_2212.10511.pdf` | Parametric vs non-parametric memory boundaries. |

### Useful For Implementation Details

| Priority | Paper | Local file | Why it matters |
|---|---|---|---|
| P1 | Kudo and Richardson, "SentencePiece..." | `references/papers/related/sentencepiece_1808.06226.pdf` | Tokenizer behavior behind Engram compression/canonicalization. |
| P1 | Zhang and Sennrich, "RMSNorm" | `references/papers/related/rmsnorm_1910.07467.pdf` | Engram's gate normalization and lightweight deployment. |
| P1 | Cormode and Muthukrishnan, "Count-Min Sketch" | `references/papers/related/count_min_sketch_2005.pdf` | Multi-hash sketch background; not the signed Engram estimator. |

### Already In Corpus And Still Relevant

| Priority | Paper | Local file | Why it matters |
|---|---|---|---|
| P0 | Sun et al., "Retentive Network" | `references/papers/core/retnet_2307.08621.pdf` | Main backbone. |
| P0 | Wu et al., "Memorizing Transformers" | `references/papers/related/memorizing_transformers_2203.08913.pdf` | External KV memory and non-differentiable retrieval. |
| P1 | Katharopoulos et al., "Linear Attention" | `references/papers/related/linear_attention_2006.16236.pdf` | Recurrent attention/state update background. |
| P1 | Wang et al., "DeepNet" | `references/papers/related/deepnet_2203.00555.pdf` | Residual scaling/stability background. |

## 4. Papers Still Worth Fetching Later

The Engram PDF also references several likely-useful sources that are not yet
mirrored locally:

- SuperBPE: for tokenizer compression and vocabulary scaling.
- RULER / LongPPL / NIAH-related long-context benchmarks: for evaluating sparse
  attention residual plus Engram effects.
- DeepSeek-R1: only if we seriously pursue the RL-training angle.
- REALM/RAG/Mention Memory: for external memory and retrieval comparisons.

These are second wave. The first wave above is enough to keep the current proof
work grounded.

## 5. Net Answer

Gemini's non-RL architecture claims are mostly consistent with the current plan.
The only meaningful drift risk is over-promoting Engram from:

```text
conditional static/semi-static memory
```

to:

```text
online RL-updated, correctness-preserving knowledge system
```

That latter claim is not in the Engram paper and needs separate proof plus
experiments.

## 6. Recorded Follow-Up Constraint

For future proof work, treat Gemini's online update and temporary cache ideas as
the following conditional contract:

```text
Engram session/online updates may be used only when update norm, active update
count, namespace separation, hash capacity, residual scale, and downstream
margin are explicitly controlled.
```

This is now formalized in:

```text
docs/proofs/10-engram-update-stability-proof.md
```
