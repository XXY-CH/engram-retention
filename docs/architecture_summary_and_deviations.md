# Final Architectural Blueprint and Proof Realignment

Created: 2026-05-03
Updated: 2026-05-04 by Codex to separate paper-backed claims from project extensions.

## 1. The Finalized Architecture (The "Four-in-One" System)

The architecture has evolved from a naive combination into a bounded, orthogonal
research scaffold. It should be presented as a conditional small-state memory
architecture, not as a universal Transformer replacement or global `O(1)`
long-context guarantee. MoE is deferred for engineering isolation, not rejected
as a final direction: the current Dense-first code/proof target isolates RetNet,
AttnRes, Engram, and snapshot behavior before sparse expert routing is
reintroduced.

### A. Sequence/Time Axis: RetNet + Optional Milestone Gate (The Conveyor Belt)
*   **Mechanism**: Linear sequence modeling via recurrent retention.
*   **Role**: Handles local syntax and short-term smoothing. Ensures $O(1)$ inference cost and zero KV-cache scaling per token.
*   **The Enhancement**: **Milestone-Triggered Pass-Through Gating**. The model may be trained to emit explicit `<MARK_THOUGHT>` / `<CRITICAL_ASSUMPTION>` tokens. Those tokens condition the retention gate for a protected subspace, slowing decay for the current logical state. This is a project extension, not a RetNet paper result. It must include a finite milestone budget or weak age penalty; unconditional permanent $\gamma=1$ causes state saturation.

### A2. Exact Token-Time Axis: Milestone Snapshot Readout (The Protected Notes)
*   **Mechanism**: A tiny bounded cache stores hidden vectors adjacent to explicit milestone markers, then later reads them with a small attention/logit-bias branch.
*   **Role**: Handles high-entropy exact recall that RetNet recurrence cannot reliably preserve. This is the restored token-time sparse-anchor idea, now clearly separated from Kimi depth-axis AttnRes.
*   **Boundary**: It costs `O(B_time)` in the number of stored snapshots. It should be used only for exact milestone recall, not as a full Transformer KV cache replacement.

### B. Depth/Layer Axis: Block Attention Residuals (The Time Machine)
*   **Mechanism**: Softmax attention across block-level representations (summaries of earlier layers).
*   **Role**: Prevents depth-wise feature dilution (the PreNorm problem). It lets deep layers retrieve earlier layer/block features without multiplying through every intermediate layer Jacobian.
*   **Boundary**: Kimi AttnRes is depth-axis, not a token-time memory cache. It can help long reasoning only through the quality of the RetNet states and layer/block summaries it reads.
*   **The Enhancement**: **Distance-Penalized Attention Residuals (ALiBi-style Bias)**. As a project extension, the depth-source attention score may include a small source-age or block-distance penalty. This creates a soft recency bias among depth/milestone summaries without deleting old summaries. It is not yet paper-backed by Kimi AttnRes and must be treated as an experimental knob.

### C. Knowledge/Width Axis: Hard Engram (The Static Encyclopedia)
*   **Mechanism**: Deterministic N-gram hashing to an offloadable, static key-value table. 
*   **Role**: Strips the burden of memorizing rote facts (e.g., specific names, dates) from the backbone. Operates strictly in $O(1)$ time.
*   **The Enhancement**: **Tokenizer / Entity Canonicalization, With Caution**. Paper-backed tokenizer compression can collapse superficial variants. Entity canonicalization is a future extension and must not be sold as full semantic retrieval. Logit-prompting is another possible future mode, but the active proof still covers residual Engram injection, not final-layer logit bias.

### D. Channel/Feature Axis: Dense FFN First, MoE Later (The Processor)
*   **Mechanism**: Standard dense feed-forward networks (e.g., SwiGLU).
*   **Role**: Nonlinear feature transformation in the first implementation. Dense FFN is the control baseline used to avoid confounding memory-path experiments with routing collapse, expert load balancing, and all-to-all communication.
*   **Future Extension**: MoE remains a channel/capacity-axis extension. It should be reintroduced only after the Dense system has measured branch norms, memory bandwidth, routing-sensitive residual variance, and module-drop contributions.

---

## 2. Deviations in Current Proofs (The "Mathematical Debt")

The current documents in `docs/proofs/` contain both active proofs and archived/appendix proofs. The major historical deviations are now mostly labeled, but the architecture has acquired new proof obligations:

1.  **The "A-Time" vs. "A-Depth" Mix-up (Proofs 01, 02, 06, 08):**
    *   **Current State**: The A-series proofs try to show that *Attention Residuals* solve the *time-axis* exponential decay of RetNet.
    *   **The Reality**: The Kimi paper defines Attention Residuals as a *depth-axis* mechanism (layer-to-layer). The math in the A-proofs applies to a *Sparse Token Anchor Cache*, not Kimi's Block AttnRes.
    *   **Current Correction**: `17-depth-attnres-non-dilution-proof.md` handles A-Depth. The older A-Time sparse-anchor proofs are marked appendix. A separate milestone-gate proof is still needed if gating becomes active.
2.  **The "Budgeted Gate" Obsession (Proofs 05, 07, 09, 14):**
    *   **Current State**: The B-series proofs meticulously calculate how to fit selected tokens into a limited cache budget $B$.
    *   **The Reality**: With the shift to depth-wise Block AttnRes, the old sparse KV-anchor budget is no longer active. But milestone gating still needs a different budget: false-positive control, protected-subspace capacity, and anti-spam penalties.
    *   **Required Fix**: Keep B-series as appendix for optional anchors. Add a new milestone-gate theorem if `<MARK_THOUGHT>` is adopted.
3.  **The Engram EMA Code Anomaly (Codebase vs. Proofs):**
    *   **Current State**: Proofs C1-C3 correctly analyze hash collisions. However, `src/layers/engram.py` implements an Exponential Moving Average (EMA) state-fusion.
    *   **The Reality**: The code is completely disconnected from both the original DeepSeek paper and our C-proofs.
    *   **Required Fix**: `engram.py` must be rewritten as a paper-faithful N-gram hash lookup module before it is used as evidence.

4.  **The Milestone Gate Proof Gap:**
    *   **Current State**: The active Dense theorem excludes token-time gates/anchors.
    *   **The Reality**: If long context depends on `<MARK_THOUGHT>`-triggered pass-through, this is now a first-class component and requires its own conditional theorem.
    *   **Required Fix**: Prove a bounded cumulative-decay condition, not an unconditional $\gamma=1$ forever claim.
5.  **The Exact Recall Snapshot Gap:**
    *   **Current State**: Stress diagnostics show milestone gating alone preserves amplitude but does not guarantee retrieval of high-entropy passwords from the saturated RetNet state.
    *   **The Reality**: Exact recall requires a queryable token-time value path. The smallest current implementation is `MilestoneSnapshotReadout`, a bounded sparse-anchor cache plus readout branch.
    *   **Required Fix**: Treat this as a separate optional theorem and experiment: prove/query-test recall as a function of snapshot budget, marker recall, readout attention mass, and logit-bias scale.

---

## 3. The Identified Vulnerabilities (The "Potholes")

Even with the theoretical framework corrected, executing this architecture requires navigating four deadly engineering traps:

### The "Explosive" Mathematical Traps (Will cause `NaN` or crashes)
1.  **Same-Layer Composition Collision**: Injecting high-variance Engram hash vectors and AttnRes depth features into the same residual stream simultaneously will destroy the dense FFN's input manifold. 
    *   *Fix*: **Interleaved Placement** (e.g., Engram at layers 2/15, AttnRes at layers 6/12/18) and **LayerScale** initialization ($1e-4$) for soft landing.
2.  **Gating Instability**: Data-dependent pass-through gates are notoriously difficult to train, prone to locking fully open (gradient explosion) or fully closed (forgetting).
    *   *Fix*: Mamba-style negative bias initialization (forcing the gate mostly closed initially) and RL-driven discrete `<MARK_THOUGHT>` tokens to enforce discrete, rather than continuous, gating decisions.

### The "Silent" Engineering Traps (Will cause poor performance or slow training)
3.  **The "Dead Weight" Syndrome**: Because the safety initializations (LayerScale=$1e-4$, Gate Bias=-3.0) heavily suppress Engram and AttnRes early on, the dense backbone might optimize itself to ignore them entirely. The advanced modules become dead weight.
    *   *Fix*: Layer-wise learning rate scaling or strict two-stage training curriculums to explicitly force reliance on these modules.
4.  **The Semantic Miss Trap (Hash Brittleness)**: Despite surjective compression, strict N-gram hashing will inevitably miss on novel phrasing. If the model is too small to rely on its dense weights, reasoning will collapse when Engram fails.
    *   *Fix*: Do not treat Engram as a replacement for model capacity. It must be framed strictly as a *top-level Logit Prompter* (biasing output probabilities) or a static encyclopedic fallback, never as a load-bearing pillar for dynamic reasoning.
5.  **The Milestone Spam / Logic Cliff Trap**: If `<MARK_THOUGHT>` is cheap, the model may spam it and saturate the protected state. If it is too expensive, the model may skip a crucial milestone and the later reasoning chain falls off a cliff.
    *   *Fix*: Use a milestone budget, false-positive penalty, minimum-distance regularizer, and separate recall/precision evaluation for milestone labels.
6.  **The Distance-Penalty Tuning Trap**: A large recency penalty makes old-but-critical assumptions unreachable; a tiny penalty fails to prevent stale milestone clutter.
    *   *Fix*: Treat the penalty as a measured hyperparameter with explicit old-assumption recall tests, not as a guaranteed theorem.
7.  **The Snapshot Overfitting Trap**: A direct snapshot-to-logit branch can solve synthetic exact-copy tasks while becoming a brittle shortcut on natural language.
    *   *Fix*: Report it separately from RetNet/AttnRes, keep the snapshot budget tiny, and ablate residual-only versus logit-bias readout on non-copy reasoning tasks.
