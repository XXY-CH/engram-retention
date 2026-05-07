# Dense Baseline Conditional Feasibility Theorem

Created: 2026-05-03

Status: active phase-1 global theorem after correcting Attention Residual
terminology and deferring MoE from the Dense-first baseline.

## 0. Architecture Being Proven

The active target is:

```text
Dense RetNet + Block AttnRes + Engram
```

with optional future extensions explicitly excluded from the theorem:

```text
deferred now: MoE expert routing
excluded now: token-time Sparse Anchor Residual / dynamic KV anchor cache
excluded now: milestone-triggered gated retention, unless proof 20 is adopted
```

The active resource statement is:

```text
per-token backbone state: O(1) RetNet recurrent state
depth residual state:     O(N_depth d) Block AttnRes summaries
knowledge lookup:         O(K) Engram hash-head lookups
channel mixing:           Dense MLP/FFN, phase-1 no expert routing
```

`N_depth` is the number of stored depth/block summaries, `d` is hidden width,
and `K` is the number of Engram hash heads.

## 1. Load-Bearing Assumptions

### R: RetNet Backbone

The recurrent RetNet state is bounded or contractive in the norm used for
analysis:

```text
||Gamma||_op <= gamma < 1
```

or the implementation provides an equivalent bounded-state invariant through
normalization/scaling.

This proves only small-state streaming for the backbone. It does not prove exact
retention of arbitrary distant facts.

### D: Block AttnRes Depth Readout

For a useful earlier layer/block source `b_i` and later layer `l`, Block AttnRes
assigns sufficient attention mass:

```text
beta_{i->l} >= epsilon_depth > 0
```

and the useful gradient direction is not cancelled by competing value/key/score
terms:

```text
value-path lower bound > cancellation upper bound
```

The local proof is `docs/proofs/17-depth-attnres-non-dilution-proof.md`.

### E: Engram Hash Retrieval

For active Engram keys `N`, slot count `M`, hash heads `K`, residual scale
`alpha`, and downstream margin `mu`, collision/update noise remains below the
margin conditions in proofs 03/04/10.

Representative signed-noise retrieval condition:

```text
L_G |alpha| R sqrt((N-1)/(K M delta_C)) < mu
```

Session updates and direct slot edits are not assumed safe unless the stricter
C3 update conditions hold.

### F: Dense Channel Mixing

The MLP/FFN block is Dense and per-token. It introduces ordinary Jacobian
boundedness assumptions:

```text
||J_F||_op <= L_F
```

It does not introduce routing, top-k expert selection, load-balancing loss, or
expert all-to-all communication.

## 2. Main Theorem

Let `Y` be a finite protected probe set. Suppose:

1. RetNet recurrent states remain bounded under assumption R.
2. For every protected depth source needed by later layers, Block AttnRes
   satisfies assumption D with failure probability at most `delta_D`.
3. Engram retrieval/update perturbations satisfy assumption E with failure
   probability at most `delta_E` per protected probe.
4. Dense MLP/FFN Jacobians and normalization Jacobians remain bounded and do not
   erase the useful direction.
5. Engram and Block AttnRes residual injections satisfy the composition guard in
   `docs/proofs/19-residual-injection-composition-guard.md`, or an empirically
   equivalent norm/scale bound.

Then on the joint event, the model has:

```text
small-state sequence streaming through RetNet
direct non-diluted depth-wise access through Block AttnRes
bounded-noise external/static memory through Engram
no MoE routing instability
```

and the joint success probability satisfies:

```text
Pr(success) >= 1 - delta_D - |Y| delta_E
```

up to any additional implementation-measured normalization/Jacobian failure
probability.

## 3. What This Theorem Does Not Claim

It does not claim:

```text
global O(1) exact long-context memory
MoE-scale capacity expansion in phase 1
token-time exact anchor recall
milestone-triggered retention without proof 20 conditions
RL gate convergence
safe online Engram editing without C3 conditions
transfer of Engram paper benchmark gains to RetNet
```

Those are separate hypotheses or future extensions.

## 4. Proof Sketch

1. RetNet gives small recurrent sequence state under R. The proof relies on the
   bounded recurrence, not on perfect long-context preservation.
2. Proof 17 shows that if a previous depth source receives Block AttnRes mass
   at least `epsilon_depth`, then the value path from later layer to that source
   contains no product over all intermediate layer Jacobians. Therefore depth
   access does not exponentially decay with layer distance under D.
3. Proofs 03 and 04 bound Engram hash collision noise and convert that noise
   into a downstream margin-safe perturbation condition.
4. Proof 10 extends Engram safety to controlled updates/session branches under
   stricter update-norm and namespace assumptions.
5. Dense MLP/FFN contributes ordinary bounded Jacobian factors only. Since MoE
   is deferred in phase 1, no routing/budget/expert-load event is required here.
6. A union bound over depth-readout failures and Engram protected-probe failures
   gives:

```text
Pr(success) >= 1 - delta_D - |Y| delta_E
```

This is the active Dense closure. The older `r q d > |Y|delta_C + delta_B`
closure belongs to optional token-time Sparse Anchor Residuals, not to Kimi
Block AttnRes.

## 5. Residual Composition Guard

The theorem previously left one local gap:

```text
Dense RetNet block Jacobian / normalization stability with Block AttnRes and
Engram residual injection composed in the same layer stack.
```

That gap is now converted into an explicit implementation contract in:

```text
docs/proofs/19-residual-injection-composition-guard.md
```

The guard requires small independent LayerScale/ReZero-style branch scales,
default-closed gates when gates exist, preferably interleaved placement, and
branch-norm logging. It is not a universal theorem; it is the active safety
condition for the first Dense implementation.

## 6. Optional Extension Hooks

Two later extensions now have separate proof files:

```text
docs/proofs/20-milestone-triggered-retention-gate.md
docs/proofs/21-distance-penalized-attnres.md
```

They are not part of this baseline theorem until the implementation explicitly
enables milestone gates or distance-penalized AttnRes.
