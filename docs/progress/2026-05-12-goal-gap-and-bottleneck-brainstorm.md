# Goal Gap And Bottleneck Brainstorm

Date: 2026-05-12

Status: research direction update after the current memory-path checkpoint.

## 2026-05-12 5.5pro ARC Note Review

The reviewed document, `ARC 条件小状态记忆.docx`, proposes an Addressed
Reasoning Controller with four parts:

```text
Memory Manager
Write Gate
Read Policy
Reasoning Fusion
```

The useful contribution is architectural: ARC names the missing control surface
between RetNet, TokenCopyBuffer, snapshots, Engram, and Block AttnRes.

The wording needs downgrading before it enters the paper. The note says ARC can
provide theoretical guarantees and deployment foundations, but the current
project can only treat ARC as a conditional target design until the following
are measured:

```text
capture -> keep -> address -> fuse -> decision
```

The first code-level debugging step is therefore not a full ARC implementation.
It is exposing TokenCopyBuffer alignment diagnostics:

```text
token_copy_weights
token_copy_valid
token_copy_pos_ids
```

These diagnostics make the current query-side addressing bottleneck measurable.

## Target

The long-term target is not only "long context". The target is the conjunction:

```text
small parameter count
+ strong reasoning
+ long-context stability
+ low compute and memory cost
```

This conjunction is stricter than any one metric. A model that solves synthetic
Needle but cannot reason does not satisfy it. A model that reasons only by
adding a large KV cache does not satisfy it. A model that is cheap only because
it drops recall does not satisfy it.

## Current Position

The project is currently at the mechanism-checkpoint stage.

| Axis | Current evidence | Current gap |
|---|---|---|
| Small parameters | The dense scaffold is toy-scale, about hundreds of thousands of parameters in local diagnostics. | No 10M/100M+ scaling curve yet. |
| Long-context recall | Needle-style exact-copy paths work inside the trained range, with TokenCopyBuffer and snapshot-style value paths. | 1024-trained models do not yet robustly extrapolate to 4096/8192. |
| Strong reasoning | XOR and xor-final diagnostics show toy recurrence behavior. | No compositional, multi-hop, natural-language, or proof-style reasoning evidence. |
| Low compute cost | RetNet recurrent state, bounded copy/snapshot buffers, and CPU Engram placement support the intended resource split. | Runtime is not yet competitive; MPS long-sequence training/eval is slow. |
| Architecture clarity | RetNet, Engram, Block AttnRes, snapshot, and TokenCopyBuffer roles are now separated. | The readout/controller policy is still under-specified. |

The strongest honest statement is:

```text
The scaffold has early evidence for bounded auxiliary memory paths, but not yet
for length-invariant reasoning or general long-context superiority.
```

## Existing Bottlenecks

### 1. Query-side addressing is the immediate blocker

TokenCopyBuffer captures raw token embeddings, but readout still depends on a
final hidden query aligning to the correct buffer slot. The latest length-scaling
pressure suggests that the model can learn the 1024 distribution without
learning a length-invariant addressing algorithm.

Symptom:

```text
curriculum helps at the trained target length, but does not automatically give
stable 4096/8192 extrapolation.
```

Likely failure mode:

```text
source memory exists, but the query state no longer selects the right slot after
more filler tokens or unfamiliar answer positions.
```

### 2. Curriculum is diagnostic, not a solution

Training at longer lengths can activate the mechanism at those lengths, but this
is a weak answer to the research goal. It increases compute cost and does not
prove the model has learned a reusable program.

Curriculum should be used to answer:

```text
Can this architecture represent the behavior if trained on that regime?
```

It does not by itself answer:

```text
Does the architecture generalize out of distribution by length?
```

### 3. Exact-copy evidence is not reasoning evidence

Needle-style tasks establish high-entropy token-time recall. They do not
establish reusable intermediate reasoning. The project must avoid collapsing:

```text
copy recall != factual lookup != reasoning state reuse
```

Each capability needs separate diagnostics and module-drop evidence.

### 4. Engram is still mostly closed-domain lookup

The current hard N-gram Engram path is promising for static or semi-static
tables, but it does not solve semantic aliasing, paraphrase, entity resolution,
or open-key generalization. A stronger knowledge claim needs canonicalized keys,
namespace control, collision telemetry, and causal module-drop evidence.

### 5. Systems cost is not yet proven

The intended systems story is good:

```text
RetNet streams ordinary tokens.
Bounded buffers keep rare high-value information.
Engram stores static patterns off accelerator.
Block AttnRes reuses depth states.
```

But the empirical cost story is still thin. MPS becomes slow at long sequence
lengths, and CPU Engram offload is currently synchronous. The architecture needs
latency, throughput, peak memory, and table-residency measurements tied to
accuracy.

## Bottlenecks Likely To Appear Next

| Future bottleneck | Why it will appear | What would expose it |
|---|---|---|
| Controller collapse | A learned capture/readout controller may close everything or open too much. | Capture recall, false-positive rate, buffer occupancy, and module-drop deltas. |
| Shortcut overfitting | Snapshot or copy logits can solve synthetic exact-copy while bypassing reasoning. | Tasks where copied facts must be transformed or composed before answer. |
| Buffer budget pressure | Fixed `K` works for one marked fact but not many independent facts. | Multi-needle and multi-query tasks with variable number of relevant facts. |
| Engram collision hot spots | Natural text is not uniform; frequent N-grams and aliases collide structurally. | Bucket load histograms, collision covariance, and high-frequency phrase stress tests. |
| Residual-path interference | Engram, AttnRes, snapshot, and copy logits share the residual/logit stream. | Branch-norm ledgers, pairwise module drops, and scale sweeps. |
| Scaling discontinuity | Toy-scale mechanisms may not behave the same at 10M/100M parameters. | Parameter scaling curves under matched data and compute. |
| Baseline embarrassment | A simple selected-KV or memory-token baseline may solve the same tasks cheaper. | Iso-parameter and iso-FLOP comparisons. |
| Training signal sparsity | Rare milestones and rare queries provide weak gradients. | Learning curves under oracle capture, auxiliary retrieval loss, and no auxiliary loss. |

## Solution Brainstorm

### A. Length-invariant copy controller

Build an explicit small state machine around copy readout:

```text
on milestone: store the previous `password_len` source tokens
on QUERY: enter answer mode
on each answer step: read slot offset 0, then 1, then 2, ...
```

This is the most direct test of whether the current failure is query addressing.
It is not the final architecture for natural language, but it is the cleanest
diagnostic. If it fixes 1024 -> 8192, the next theorem should focus on learned
controllers approximating this state machine.

Trade-off: more hand-coded structure, less elegance. The value is diagnostic
clarity.

### B. Relative-position readout instead of absolute-position readout

Make buffer keys and queries depend on relative roles:

```text
source role: token before milestone, offset -3/-2/-1
query role: token after QUERY, answer offset 0/1/2
```

This avoids expecting the model to extrapolate absolute positions from 1024 to
8192. It also matches the task's real invariant: the password-to-answer relation
is role-relative, not absolute-position-relative.

Trade-off: still synthetic-task-shaped, but less hand-coded than a full pointer
state machine.

### C. Auxiliary cache-query contrastive loss

Add a training target that directly teaches which buffer slot the answer
position should attend to. This attacks the bottleneck in the attention weights
instead of waiting for masked LM loss to discover the alignment indirectly.

Useful metrics:

```text
cache attention top-1 accuracy
cache attention entropy
correct-slot logit margin
base-vs-copy logit margin
```

Trade-off: adds supervision and training complexity. It may be necessary if the
natural gradient signal is too sparse.

### D. Oracle-to-learned capture/readout ladder

Use oracle capture and oracle readout first to estimate headroom:

```text
1. oracle capture + oracle readout
2. oracle capture + learned readout
3. learned capture + oracle readout
4. learned capture + learned readout
```

This separates failure causes. Without this ladder, a bad run can be blamed on
capture, retention, readout, optimization, or margins all at once.

Trade-off: slower experimental program, but much clearer evidence.

### E. Reasoning-state tasks before natural language

Create tasks where the model must store intermediate conclusions, not raw answer
tokens:

```text
premise A appears early
premise B appears later
model must combine A and B after distractors
answer is not copied verbatim from either premise
```

Examples:

```text
delayed variable binding
multi-hop key-value chains
symbolic rule application
short proof-state reuse
```

Trade-off: harder to debug than Needle, but necessary for the "strong reasoning"
part of the target.

### F. Canonicalized Engram path

Keep hard Engram lookup, but add a canonicalization layer before hashing:

```text
surface phrase -> canonical key -> namespace/hash -> Engram slot
```

Near-term versions can be synthetic:

```text
alias tables
case/format normalization
entity ID remapping
domain namespaces
```

Longer-term versions can use learned projection or LSH. The key is to measure
hit rate and module-drop deltas rather than assume lookup correctness.

Trade-off: improves knowledge lookup, but does not solve reasoning by itself.

### G. Hot/cold Engram cache

Extend CPU table placement into a real systems hierarchy:

```text
hot slots: accelerator
warm slots: CPU memory
cold shards: disk/NVMe
```

Measure:

```text
hit rate
transfer overhead
prefetch success
accuracy with/without hot cache
```

Trade-off: systems work can distract from core architecture unless paired with
accuracy metrics.

### H. Selected-KV baseline as a friendly adversary

Implement or import a simple selected-KV/memory-token baseline. Treat it as the
baseline to beat, not as an enemy. If selected-KV solves the same sparse recall
problem with less complexity, the research claim must shift.

Trade-off: may weaken the story, but prevents building a beautiful mechanism
that loses to a simpler cache.

## Recommended Next Sequence

### Step 1: Prove or falsify the query-addressing diagnosis

Run a 1024 -> 2048/4096/8192 comparison:

```text
current TokenCopyBuffer
vs explicit answer-offset pointer readout
vs relative-role readout
```

Success criterion:

```text
train at 1024, eval at 8192, stable exact match without training at 8192.
```

### Step 2: Add cache-query telemetry

Every copy/snapshot run should emit:

```text
correct slot attention
attention entropy
copy logit margin
base logit margin
buffer valid count
module-drop exact match
```

This turns vague "OOD query" language into measurable failure modes.

### Step 3: Move from copy to reasoning-copy

After length-invariant copy works, add a task where the stored facts must be
composed. Do not jump straight to natural language benchmarks until the
mechanism survives controlled composition.

### Step 4: Start resource-accounted baseline comparisons

Compare against:

```text
RetNet-only
Transformer
selected-KV / memory-token baseline
current hybrid
```

Report:

```text
parameters
peak memory
latency
eval EM/loss
effective cache/buffer size
```

## Update Needed In The Paper

The paper should be updated with three facts:

1. TokenCopyBuffer is now a first-class exact-copy mechanism, not just an
   implementation detail.
2. Length extrapolation is the current negative result: training at 1024 does
   not yet imply stable 4096/8192 behavior.
3. The immediate bottleneck is query-side addressing into bounded memory, not
   only RetNet decay or position-key encoding.

This update should make the paper more credible, not more pessimistic. A clear
negative result tells us where the architecture must become more algorithmic.
