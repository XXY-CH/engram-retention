# Canonicalized Engram Retrieval Theorem

Created: 2026-05-04

Status: future theorem track for expanding Engram from exact lexical lookup
toward knowledge retrieval while preserving the hard-hash/offload boundary.

## Claim Boundary

Paper-faithful Engram is deterministic N-gram hash lookup. It is not semantic
RAG. To claim knowledge retrieval enhancement, we need a measurable key
canonicalization and retrieval-margin theorem.

## Target Statement

Let a fact query have canonical key:

```text
c = Canonicalize(surface_form)
```

and Engram retrieve:

```text
e = E[Hash(c)]
```

If:

1. Canonicalization hits the intended fact key with probability at least `r_k`.
2. Hash collision noise has norm at most `epsilon_hash` with probability
   `1 - delta`.
3. Engram residual or logit contribution gives the correct fact margin at least
   `m_e`.
4. Downstream model margin loss from perturbation is below `m_e`.

Then the factual prediction remains correct with probability at least:

```text
r_k (1 - delta)
```

up to downstream margin failures.

## Required Evidence

The synthetic diagnostic must make Dense memorization insufficient:

```text
large key space
small d_model
train/eval key split
module-drop(engram) causes a measurable drop on seen facts
unseen-key split is reported separately and not counted as memorization success
```

## Non-Claim

Hard Engram does not solve fuzzy semantic lookup unless the canonicalizer maps
the fuzzy surface forms to the same key. That canonicalizer is a separate module
with its own error rate `1 - r_k`.

## Current Evidence Boundary

The 2026-05-04 harder `alien_static` diagnostic gives weak positive evidence for
seen static facts:

```text
train-key eval:
  ours_nosnapshot EM = 0.992
  no-Engram EM      = 0.969
  RetNet EM         = 0.922
```

But test-key evaluation remains:

```text
test-key EM = 0.0
```

Therefore the current implementation supports only a limited seen-fact memory
claim. It does not support unseen factual generalization or fuzzy retrieval.
