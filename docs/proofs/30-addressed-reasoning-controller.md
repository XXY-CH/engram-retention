# Conditional Theorem: Addressed Reasoning Controller

Created: 2026-05-12

Status: target-design proof contract after reviewing the ARC research note. This
document does not claim that ARC is implemented or empirically validated. It
states the conditions a small controller must satisfy before it can be treated
as the unifying read/write/fusion mechanism for bounded memory.

## 0. Motivation

The current architecture has useful memory paths but no single proof surface for
the control decisions:

```text
when to write
what to write
where to write
when to read
which slot to read
how to fuse the result
```

RetNet provides the streaming state. TokenCopyBuffer and milestone snapshots
provide bounded exact or high-entropy value paths. Engram provides static or
semi-static lookup. Block AttnRes provides depth reuse. ARC is the proposed
small controller that routes among those paths without falling back to a full
Transformer-style KV cache.

## 1. Controller Interface

At token step `t`, ARC observes a bounded feature vector:

```text
u_t = phi(h_t, token_t, role_t, r_t, telemetry_t)
```

where `h_t` is the current hidden state, `role_t` is a discrete or learned role
feature such as milestone/query/answer-offset, `r_t` is a bounded recurrent
controller state, and `telemetry_t` contains local signals such as surprisal,
buffer occupancy, and branch margins.

ARC emits:

```text
w_t in {none, token, hidden, fact, depth}
rho_t in Delta(K_max) or top-k over bounded memory
f_t in [0, 1]^J
```

where `w_t` is the write action, `rho_t` is the bounded read distribution, and
`f_t` are fusion gates for the available memory branches.

The key resource constraint is:

```text
K_max, B_max, H_max, and controller state size are fixed by policy.
```

## 2. Budget Proposition

If ARC's controller state has fixed size `S_A`, reads at most `K_max` snapshot or
copy slots, reads at most `B_max` depth sources, and performs at most `H_max`
Engram probes per token, then ARC changes only the constant factor of the
RetNet-style scan:

```text
per-token cost = O(S_A + K_max d + B_max d + H_max d)
sequence cost  = O(n (S_A + K_max d + B_max d + H_max d)).
```

If all caps are independent of sequence length `n`, the sequence cost is
linear in `n`. If any cap grows with `n`, the claim must be restated as
`O(n K(n))`, `O(n B(n))`, or `O(n H(n))`.

## 3. Address Margin Condition

Let `s*` be the correct memory slot for the current decision and let ARC assign
slot scores:

```text
a_j = q_t^T k_j + b_role(role_t, meta_j) - c cost_j.
```

A sufficient condition for correct top-1 read is:

```text
a_{s*} >= max_{j != s*} a_j + Delta_addr
```

for `Delta_addr > 0`. For softmax readout, a sufficient mass condition is:

```text
Delta_addr >= tau log((K_max - 1)(1 - epsilon) / epsilon)
```

which gives read mass at least `1 - epsilon` on the correct slot.

This is the mathematical version of the current query-side bottleneck: it is not
enough for the memory to contain the source value. ARC must create a role/content
address margin that survives length extrapolation.

## 4. Fusion Margin Condition

Let `r_t` be the read value produced by ARC and let `z_base` be the logits from
the backbone before memory fusion. The memory branch contributes:

```text
z = z_base + lambda_f W_f r_t.
```

For target token `y*`, a sufficient decision condition is:

```text
(z[y*] - max_{y != y*} z[y]) >= m_decision > 0.
```

Equivalently, the memory logit advantage must exceed any wrong-token advantage
created by the base model. This reuses the TokenCopyBuffer and snapshot logit
margin contract; ARC only decides which value is read and how strongly it is
allowed to affect the decision.

## 5. Stability And Collapse Conditions

ARC must avoid two degenerate policies:

```text
write nothing / read nothing
write everything / read everything
```

A bounded controller objective should therefore track:

```text
capture recall
false-positive write rate
buffer occupancy
read entropy
correct-slot attention mass
fusion gate norm
module-drop decision delta
```

For a learned gate, the proof obligation is not "the gate will learn." The
condition is:

```text
expected utility margin > sparsity price + estimation error.
```

Only under that condition does a budgeted write/read policy select useful memory
items rather than collapsing to closed or open behavior.

## 6. Measurement Contract

Every positive ARC claim should report:

```text
capture -> keep -> address -> fuse -> decision
```

with separate metrics for each arrow. In the current codebase, the first
low-risk step is to expose TokenCopyBuffer alignment diagnostics:

```text
token_copy_weights
token_copy_valid
token_copy_pos_ids
```

These diagnostics turn the vague claim "query OOD" into measurable correct-slot
attention and slot-margin failures.

## 7. Non-Claim

ARC is not a free theorem and not a hidden Transformer. If ARC attends over all
past tokens, the design has reverted to KV-cache attention. If ARC's memory caps
grow with context length, the budget claim must expose that growth. If ARC
solves only exact-copy Needle tasks, it is not evidence of strong reasoning.

The defensible claim is narrower:

```text
A bounded ARC may preserve linear-time scanning while giving the architecture a
measurable controller surface for selective write, sparse read, and gated fusion.
```

