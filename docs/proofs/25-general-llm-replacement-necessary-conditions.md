# Necessary Conditions For A General LLM Replacement Claim

Created: 2026-05-04

Status: scope-tightening theorem. This document does not prove the current
architecture is a general LLM replacement. It states what must be proven before
that stronger claim is allowed.

## 0. Motivation

The current supported claim is a bounded auxiliary-memory RetNet scaffold. To
move toward a general LLM replacement, we need to show that the hybrid system can
cover the functional roles of a Transformer/MoE baseline under explicit resource
accounting.

## 1. Resource Accounting

Let:

```text
T        = context length
d        = hidden dimension
S_R      = RetNet recurrent state size
B_time   = token-time milestone snapshot budget
N_depth  = number of depth/block summaries
K        = Engram hash heads
M        = Engram slots per head
C        = canonicalizer / tokenizer mapping size
```

The effective memory resource is not `O(1)` unless `B_time`, `N_depth`, `K`, `M`,
and `C` are treated as constants independent of task difficulty.

The honest accounting is:

```text
Memory_effective =
  S_R
  + O(B_time d)
  + O(N_depth d)
  + O(KM d) external Engram table
  + O(C) canonicalization metadata
```

## 2. Information-Theoretic Lower Bound

Consider a task family where the input contains `m` independent high-entropy
facts:

```text
Z_1, ..., Z_m
```

and the query asks for one uniformly selected fact after distractors. If each
fact has entropy `H` bits and the model must answer with error at most `epsilon`,
then any internal or external memory used to recover arbitrary selected facts
must contain mutual information:

```text
I(Memory; Z_1, ..., Z_m) >= (1 - epsilon) m H - 1
```

This is a Fano-style necessary condition. It implies:

```text
strict O(1) memory cannot support arbitrary m -> infinity exact recall.
```

Therefore a replacement claim must be resource-conditional:

```text
B_time or Engram/canonicalized memory must scale with the number of independent
facts the task distribution requires.
```

## 3. Functional Coverage Conditions

A general LLM replacement must cover at least four functions:

```text
F_seq:       local and medium-range sequence mixing
F_exact:     exact high-entropy recall
F_reason:    reusable intermediate reasoning states
F_knowledge: static and changing factual knowledge
```

The hybrid architecture assigns:

```text
F_seq       -> RetNet + Dense FFN
F_exact     -> Milestone Snapshot Readout
F_reason    -> Reasoning State Reuse + Block AttnRes
F_knowledge -> Canonicalized Engram, possibly hybrid hard/soft retrieval
```

The replacement claim is allowed only if every function has both:

```text
conditional theorem + module-drop empirical evidence
```

## 4. Baseline-Dominance Condition

Let `A_hybrid(R)` be accuracy under resource budget `R`, and let
`A_base(R)` be the best matched baseline accuracy for Transformer/Mamba/RetNet.

The strongest acceptable claim is not:

```text
hybrid is universally better
```

but:

```text
there exists a task distribution D and a resource region R such that
A_hybrid(R, D) > A_base(R, D)
while latency/memory is lower or equal.
```

For a general replacement direction, this must hold on a broad benchmark family,
not only one synthetic task.

## 5. Required Proof Upgrades

The current project needs:

```text
1. Tight concentration bounds for Engram noise.
2. Lyapunov/ISS stability for gated retention under bounded updates.
3. Reasoning-state reuse theorem with downstream decision margin.
4. Resource-recall lower bound and measured resource scaling.
5. Causal module-drop tests at every claimed capability.
```

## 6. Non-Claim

This document prevents an overclaim:

```text
RetNet + bounded snapshots + Engram is not a general LLM replacement until these
necessary conditions are satisfied.
```

It gives the ladder toward that claim.
