# Proof Rigor Audit

Date: 2026-05-11

Status: completed corrective pass over the current proof trail.

## Scope

This audit checks the proof files for over-strong claims, missing assumptions,
and wording that converts synthetic diagnostics into general mathematical
guarantees.

The active proof position remains:

```text
conditional mechanism evidence under explicit resource budgets
```

It is not:

```text
state-of-the-art long-context performance
universal LLM replacement
global O(1) exact recall
automatic training-time module activation
```

## Corrections Applied

| File | Issue | Correction |
|---|---|---|
| `25-engram-hoeffding-concentration-bound.md` | Treated a projected Hoeffding-style bound as an unconditional norm guarantee and claimed absolute network safety. | Reframed as a projected signed-noise concentration bound under idealized hash assumptions; added collision-count failure and dimension/norm caveat. |
| `26-moe-routing-lyapunov-stability.md` | Claimed MoE router collapse was mathematically ruled out by small residual injection. | Reframed as a local router-perturbation bound and listed training-time MoE stability obligations. |
| `27-snapshot-gradient-flow-dominance.md` | Claimed automatic snapshot-module awakening from gradient dominance. | Reframed as a conditional gradient-advantage note requiring capture, margin, and branch-aware optimization evidence. |
| `28-residual-scale-nonnegativity-corollary.md` | Claimed `abs()` preserves branch contribution and non-dilution bounds by itself. | Reframed as a sign-reversal guard; alignment, attention mass, and logit margin remain separate assumptions. |
| `29-token-copy-buffer-expressiveness.md` | Claimed exact recall from Cauchy-Schwarz and near-orthogonal embeddings. | Reframed as a conditional exact-copy logit path requiring capture, attention concentration, embedding self-similarity margin, and base-logit margin. |
| `phase1-validation-report.md` | Reported synthetic checkpoint results as complete validation and broad benchmark success. | Reframed as a mechanism checkpoint with local diagnostic evidence. |
| `src/layers/token_copy_buffer.py` | Docstring implied unconditional exact recall. | Updated to conditional exact-copy wording tied to alignment and logit-margin conditions. |
| `src/layers/engram.py` | Implementation did not expose the host-memory/offload boundary discussed in the proof trail. | Added explicit `table_device` support and table memory/device introspection; this is CPU/host offload, not full NVMe paging. |

## Files Left As Conditional

The following files still use words such as `guarantee`, `always`, or `O(1)`,
but the usage is either explicitly conditional or part of a negative boundary:

| File | Reason left unchanged |
|---|---|
| `08-softmax-readout-mass-proof.md` | A softmax logit margin does guarantee readout mass under the stated finite-source condition. |
| `11-global-conditional-feasibility-theorem.md` | The opening explicitly says it does not prove unconditional success. |
| `15-proof-closure.md` | The strong claims appear in a "does not claim" list or in historical closure wording. |
| `22-milestone-snapshot-readout.md` | Exact prediction is conditioned on a combined logit-margin inequality. |
| `25-general-llm-replacement-necessary-conditions.md` | Strong replacement language appears as necessary conditions and lower-bound warnings, not as current capability. |
| `formal_hybrid_architecture_proofs.tex` | Already states a conditional, preliminary claim and includes a minimum bar before strong claims. |

## Reviewer Rule Going Forward

Any future proof or report that uses `exact`, `guarantee`, `validated`,
`replacement`, `dominance`, or `O(1)` must also state:

1. the resource being held fixed;
2. the capture or retrieval event;
3. the alignment or attention-mass condition;
4. the downstream margin condition;
5. the empirical obligation that would falsify the claim.

If any of these are missing, the statement should be downgraded to a hypothesis,
mechanism note, or diagnostic result.
