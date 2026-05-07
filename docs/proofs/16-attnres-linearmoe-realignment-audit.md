# AttnRes Realignment Audit And Dense-First Baseline Decision

Created: 2026-05-03
Updated: 2026-05-04 after clarifying active MoE deferral

Purpose: respond to the Gemini reassessment, re-read the local Kimi
`Attention Residuals.pdf`, record the Linear-MoE audit as future-extension
evidence, and realign the active target architecture before implementation.

## 0. Bottom Line

Gemini's second-pass correction is directionally right:

```text
Attention Residuals is a real paper-backed mechanism, and it operates along the
depth axis by attending over previous layer/block outputs.
```

But one sentence must be tightened:

```text
The previous time-axis sparse-anchor proof is not the same theorem as Kimi
AttnRes. It remains useful only as an optional Budgeted Sparse Attention Anchor
module for token-time memory.
```

Therefore the correct phase-1 architecture is no longer a vague three-part
system. It is a Dense-first multi-axis system:

```text
time/sequence axis: RetNet, optionally with data-dependent retention gates
depth axis: Kimi-style Block Attention Residuals
knowledge axis: Engram hashed conditional memory
channel axis: Dense MLP/FFN first, MoE deferred
optional exact-token axis: Budgeted Sparse Attention Anchors
```

MoE has been deferred from the active baseline because of realistic routing,
communication, memory-bandwidth, load-balancing, and implementation risks. This
is a staging decision, not a rejection of MoE. The Linear-MoE audit below remains
useful as phase-2 extension evidence.

## 1. Kimi Attention Residuals: What The Paper Actually Says

Local PDF:

```text
papers/attention-residual/Attention Residuals.pdf
tmp/pdf_extracts/local_attention_residuals.clean.txt
```

arXiv metadata:

```text
arXiv:2603.15031, submitted 2026-03-16
```

The paper's key claim is depth-wise attention over residual sources.

Evidence:

- `tmp/pdf_extracts/local_attention_residuals.clean.txt:8-18`: fixed residual
  accumulation causes hidden-state growth and dilution; AttnRes replaces fixed
  accumulation with softmax attention over preceding layer outputs; Block
  AttnRes attends over block-level representations.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:91-120`: residuals are
  both gradient highways and depth-wise information aggregation; AttnRes changes
  `h_l = sum_i v_i` into `h_l = sum_i alpha_{i->l} v_i`.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:188-205`: the paper
  explicitly states the time/depth duality and defines:

```text
h_l = alpha_{0->l} h_1 + sum_{i=1}^{l-1} alpha_{i->l} f_i(h_i)
```

- `tmp/pdf_extracts/local_attention_residuals.clean.txt:209-229`: attention
  weights are softmax over depth using a learned layer pseudo-query and
  RMSNormed keys.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:246-327`: Block
  AttnRes reduces memory/computation by attending over `N` block
  representations instead of `L` layer outputs.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:604-617`: Block
  AttnRes mitigates PreNorm dilution and yields more uniform gradient
  distribution.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:812-821`: learned
  weights show diagonal locality plus off-diagonal learned skip connections.
- `tmp/pdf_extracts/local_attention_residuals.clean.txt:1050-1078`: AttnRes is
  orthogonal to normalization/gating schemes and provides selective access to
  individual earlier-layer outputs; block structure reduces cost from `O(L^2)`
  to `O(LN)`.

## 2. What This Changes In Our Proofs

The old A-side proof was written for token-time sparse anchors:

```text
h_i at token i -> KV cache -> read at token T
```

Kimi AttnRes instead needs the depth-axis version:

```text
layer/block source i -> depth attention -> layer l
```

The mathematical shape is similar:

```text
fixed recurrence/additive path can dilute with distance/depth
attention path gives direct selected access
readout mass and dominance conditions are still required
```

But the variables must be separated:

```text
Token-time proof:
  distance = T - i
  resource = B_time anchors
  module = Budgeted Sparse Attention Anchors

Depth-time proof:
  distance = L - l
  resource = N_depth block summaries
  module = Block Attention Residuals
```

So the previous proof is not "wrong"; it is over-broadly named. It must become:

```text
A-depth: Block AttnRes prevents depth-wise residual dilution under readout mass
         and dominance conditions.

A-time: optional Budgeted Sparse Attention Anchors preserve exact critical
        token facts across sequence time.
```

## 3. Current Code Alignment

Current code is not yet aligned with the Kimi AttnRes paper.

`src/layers/attention_residual.py` currently implements:

```text
STANDARD: x + F(x)
HIGHWAY: g F(x) + (1-g) x
DEEPNET: x + alpha F(x)
DENSE: currently effectively x + F(x)
```

It does not implement:

```text
learned pseudo-query w_l
RMSNormed keys from previous layer/block outputs
softmax attention over depth sources
block summaries b_n
partial_block state
separate pre-attention and pre-MLP AttnRes sites
two-phase / cached inference schedule
```

Therefore Gemini is correct that `src/layers/attention_residual.py` needs a
major rewrite if we want Kimi-style AttnRes.

Current Engram code is also not aligned with the Engram paper:

```text
src/layers/engram.py = biological/stateful consolidation gate over retention states
Engram paper       = deterministic N-gram multi-head hash table lookup + gated residual fusion
```

That code should be treated as a prototype with the wrong name, not as the
paper-backed Engram module.

## 4. Linear-MoE Audit, Now Archived As Future Extension

Downloaded:

```text
references/papers/related/linear_moe_2503.05447.pdf
tmp/pdf_extracts/linear_moe_2503.05447.txt
```

arXiv metadata:

```text
arXiv:2503.05447
submitted 2025-03-07
last revised 2025-04-15, v2
official code: https://github.com/OpenSparseLLMs/Linear-MoE
```

Paper support:

- `tmp/pdf_extracts/linear_moe_2503.05447.txt:3-33`: Linear-MoE integrates
  Linear Sequence Modeling with MoE and reports efficiency gains with
  competitive benchmark performance.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:44-49`: LSM methods provide
  efficient training/inference and avoid maintaining KV cache.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:67-69`: LSM methods share a
  unified recurrence framework.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:196-225`: table lists RetNet,
  GLA, DeltaNet, Gated DeltaNet, Mamba, and Mamba2 as unified recurrence
  instances.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:284-325`: each Linear-MoE block
  contains an LSM layer and an MoE layer.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:332-340`: MoE uses standard
  sparse expert activation/routing.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:357-365`: hybrid Linear-MoE
  combines Linear-MoE layers with standard Transformer-MoE layers.
- `tmp/pdf_extracts/linear_moe_2503.05447.txt:616-657`: implementation supports
  LSM modules, MoE layers, Linear-MoE blocks, and adapts Qwen2/DeepSeekV2/Mixtral
  MoE architectures.

Conclusion:

```text
RetNet/linear-RNN + MoE compatibility is paper-backed.
MoE is a channel/capacity-axis component, orthogonal to RetNet's sequence mixing.
```

Phase-1 baseline correction:

```text
Do not include MoE in the phase-1 proof target.
Do not require MoE routing, expert capacity, load-balancing, or all-to-all
communication assumptions in the current Dense-first theorem.
Use Linear-MoE only if a later extension deliberately reintroduces sparse
experts.
```

## 5. Data-Dependent Pass-Through Gate

The user/Gemini proposal:

```text
critical-information gate recognizes key tokens and protects them from decay
```

is useful, but it should be positioned carefully.

RetNet's original table/formula uses a fixed scalar decay:

```text
M_s = a M_{s-1} + k_s^T v_s
```

Linear-MoE's table shows nearby LSM variants with data-dependent or
matrix/vector-dependent state updates:

```text
GLA:      M_s = diag(a_s) M_{s-1} + k_s^T v_s
DeltaNet: M_s = (I - a_s k_s^T k_s) M_{s-1} + b_s k_s^T v_s
Mamba:    exp(data-dependent terms) * M_{s-1} + ...
```

So "pass-through gating" is not paper-backed as RetNet itself, but it is
paper-near as a move from fixed RetNet toward GLA/DeltaNet/Mamba-style
data-dependent linear recurrence.

Proof implication:

```text
The old B budgeted-gate proof applies naturally to choosing which tokens receive
special preservation or exact anchor treatment. It is not required for Kimi
depth-wise AttnRes itself.
```

Implementation implication:

```text
Do not silently call this RetNet unless the recurrence remains RetNet.
Call it RetNet+Gate, Gated Retention, or a GLA/DeltaNet-inspired variant.
```

## 6. Corrected Dense Architecture Statement

The target architecture should now be stated as:

```text
RetNet / gated linear recurrence:
  sequence-time streaming backbone with small recurrent state.

Dense MLP/FFN:
  ordinary per-token channel mixing after sequence mixing, with no expert
  routing or expert-parallel communication in the active baseline.

Block Attention Residuals:
  depth-axis selective access to previous layer/block outputs, preventing
  PreNorm dilution and depth-wise feature loss.

Engram:
  static/semi-static N-gram hash-table memory with offload/prefetch and gated
  residual fusion.

Optional Budgeted Sparse Attention Anchors:
  token-time exact critical-anchor cache if synthetic tasks show RetNet/gated
  recurrence still cannot preserve rare exact facts.
```

## 7. Actionable Corrections

1. Rename the ambiguous module in docs:

```text
Attention Residual -> Block Attention Residuals when discussing Kimi depth-axis mechanism
Sparse Anchor Residual / Budgeted Sparse Attention Anchors when discussing token-time KV anchors
```

2. Rewrite `src/layers/attention_residual.py` before using it as evidence:

```text
it currently implements residual wrappers, not Kimi AttnRes.
```

3. Rewrite `src/layers/engram.py` before using it as evidence:

```text
it currently implements state consolidation, not N-gram hashed Engram lookup.
```

4. Keep MoE out of the phase-1 baseline. Preserve Linear-MoE as phase-2
   extension evidence.

5. Split A proofs:

```text
A-depth for Block AttnRes.
A-time for optional Budgeted Sparse Attention Anchors or gated retention.
```

## 8. Revised Proof Target

The stronger and cleaner proof target is:

```text
time axis:
  RetNet/gated LSM gives small-state streaming, with optional preservation gate.

depth axis:
  Block AttnRes gives direct depth-wise access and mitigates residual dilution.

knowledge axis:
  Engram gives bounded-noise offloadable lookup memory.

dense channel axis:
  Dense MLP/FFN provides standard per-token channel mixing without sparse
  expert-routing proof obligations.
```

This is more aligned with the literature than the earlier "RetNet + Engram +
attention residual" phrasing.

## 9. Follow-Up Proof Artifact

The depth-axis version is now separated into:

```text
docs/proofs/17-depth-attnres-non-dilution-proof.md
```

This prevents the old token-time sparse-anchor theorem from being mistaken for
the proof of Kimi Block AttnRes.
