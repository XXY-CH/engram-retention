# Engram Gemini Alignment Audit

Created: 2026-05-03

Purpose: compare the Gemini discussion against the downloaded Engram PDF and
check whether the current RetNet + attention residual + Engram research plan has
drifted.

Primary source:

- `references/papers/core/engram_conditional_memory_2601.07372.pdf`
- extracted notes: `tmp/pdf_extracts/engram_conditional_memory_2601.07372.clean.txt`

## High-Level Verdict

The current project direction is mostly aligned:

```text
RetNet backbone: small-memory streaming compute
Block Attention Residuals: depth-wise residual aggregation
Engram: conditional memory / offloadable lookup for static patterns and knowledge
Optional Sparse Anchor Residual: future token-time long-range anchor extension
```

The main required correction is that Engram should not be described as already
proven for RL-only online knowledge updating. The paper supports trainable
embedding tables and infrastructure-friendly lookup/offload, but does not prove
an RL-only update scheme, non-parametric slot edits, or safe online continual
knowledge updates.

## Claim-By-Claim Audit

| Gemini claim | PDF support | Verdict | Required correction |
|---|---|---|---|
| Engram uses N-gram extraction and deterministic hash lookup. | Section 2.2 defines compressed suffix N-grams, multi-head hash tables, and multiplicative-XOR hash. | Supported. | Keep. This directly supports our C1 hash/capacity proof. |
| Engram integrates by residual connection `H^(l) <- H^(l) + Y`. | Section 2.3 states residual integration after gated value projection and causal convolution. | Supported. | Keep. Our RetNet adaptation can use the same residual-stream hook. |
| Engram uses hidden-state/context-aware gating. | Section 2.3 uses current hidden state as query and retrieved memory as key/value source, with scalar gate. | Supported. | Keep, but note gate suppresses static memory; it does not prove semantic correctness. |
| Engram is topology-agnostic. | Section 2.4 says it is inherently topology-agnostic, while adaptation to multi-branch architectures still needs structural optimization. | Supported with caveat. | RetNet compatibility is plausible, but still needs a RetNet-specific insertion and normalization design. |
| Engram offloads to host memory and supports prefetch. | Section 2.5 and Figure 2 describe host offload, deterministic prefetch, overlap with compute; Table 4 measures 100B CPU-offloaded Engram overhead. | Supported. | Strongly aligns with our edge/offload direction, but H800/PCIe evidence is not Orange Pi evidence. |
| Multi-level cache hierarchy / Zipfian locality. | Section 2.5 motivates caching frequent embeddings in HBM and rare ones in host DRAM/NVMe from Zipfian N-gram locality. | Supported. | Supports our temporary/session cache idea as a system extension, not as a trained-model result. |
| Engram improves reasoning, code/math, long-context. | Abstract and experiments report BBH, ARC-Challenge, HumanEval, MATH, RULER/Multi-query NIAH gains. | Supported as experimental evidence in their MoE/Transformer setting. | Do not claim the same numbers will transfer to RetNet. Treat as motivation. |
| LogitLens/CKA show effective depth increase. | Section 6.1 reports earlier prediction convergence and Engram layer 5 aligning with about MoE layer 12. | Supported. | Keep as mechanistic motivation for shallow-layer insertion. |
| Best insertion at layer 2 or layers 2 and 15. | Experimental config uses Engram at layers 2 and 15; ablation discusses layer sensitivity. | Supported for their architecture. | For RetNet, this becomes a hypothesis: shallow insertion likely useful, but layer choice must be revalidated. |
| Freeze backbone and update only Engram with RL to keep knowledge current. | PDF says Engram embedding parameters are updated with Adam during training. RL appears only in references, not as an Engram update method. | Not directly supported. | Mark as future hypothesis. Need a separate proof/experiment for RL-only or online slot updates. |
| Non-parametric external edits to hash slots can safely inject new facts. | Paper discusses model editing literature in related work, not a proven Engram slot-editing protocol. | Not supported by this PDF. | Treat as speculative extension. Need collision, interference, and gate-safety tests before claiming. |
| Temporary online session cache can reduce repeated long-context upload. | Deterministic IDs, host offload, prefetch, and cache hierarchy support the systems premise. | Plausible extension, not directly evaluated. | Current idea is aligned but should be called session-level Engram cache / engineering hypothesis. |

## Does Our Current Model Design Drift?

No major drift. The refined design should be stated as:

```text
RetNet handles the streaming compute path.
Block AttnRes handles depth-wise residual dilution.
Engram handles static or semi-static lookup memory that can be offloaded and
prefetched, including future session-level cache tables.
Sparse Anchor Residual is optional if exact token-time anchors are needed later.
```

Fresh user-provided context should not be written into the global Engram table.
It should be processed by the RetNet recurrent stream and made available to
deeper layers through Block AttnRes summaries. Engram remains the static or
semi-static lexical/factual lookup axis.

The design would drift if we claimed:

```text
Engram alone handles continuous long-range reasoning.
Engram paper proves RL-only knowledge updating.
Engram paper proves temporary online caches preserve correctness.
Reported BBH/HumanEval gains transfer directly to RetNet.
```

## Impact On Existing Proof Chain

Existing proofs remain aligned:

- C1/C2 are now better justified by the actual Engram design: deterministic
  multi-head N-gram hash lookup, collision/polysemy noise, gated residual
  injection.
- A-depth proof remains separate: Engram is not the depth-wise residual
  aggregation mechanism; Block AttnRes is.
- A-time proofs are optional appendix material for Sparse Anchor Residuals, not
  the active baseline.
- B proofs become more important: the Engram paper's context-aware gate supports
  gating as a real mechanism, but budgeted gates for sparse residual anchors are
  only needed if the optional anchor path is reintroduced.

## Required Proof/Plan Updates

1. Add Engram PDF as a core source in `references/papers/MANIFEST.md`.
2. Treat RL Engram updating as an unproven future work item.
3. Add a future theorem target for online/session Engram caches:

```text
session cache safety = bounded hash collision/interference
                     + bounded residual injection perturbation
                     + eviction policy preserving active utility mass
```

4. Keep RetNet as the main memory-saving backbone. Engram is a conditional memory
   axis, not the replacement for RetNet's recurrent state.
