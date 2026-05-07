# Distance-Penalized Block AttnRes

Created: 2026-05-04

Status: project extension to Kimi-style Block Attention Residuals. This is not
claimed as part of the original Attention Residuals paper.

## 0. Goal

Long reasoning needs an arrow of time:

```text
recent intermediate conclusions should usually be easier to retrieve than stale
ones, but old critical assumptions must remain recoverable.
```

A hard recurrent decay can erase old information. A soft attention penalty can
prefer recent sources without deleting old sources.

## 1. Scoring Rule

For a later layer/block query `q_l` and an earlier source summary `b_n`, define:

```text
s_n = <q_l, k_n> - c d(l,n)
```

where:

```text
c >= 0
d(l,n) >= 0
```

`d(l,n)` may be depth distance, block age, or a source-age metadata term if the
implementation stores milestone summaries. In plain Kimi AttnRes it is safest
to interpret this as depth/source distance, not token-time distance.

Attention mass is:

```text
beta_n = exp(s_n) / sum_j exp(s_j)
```

## 2. Recoverability Condition

For an old but critical source `i` to receive mass at least `epsilon`, it is
sufficient that:

```text
<q_l,k_i> - c d(l,i)
>= max_{j != i} (<q_l,k_j> - c d(l,j))
   + log((m-1) epsilon / (1-epsilon))
```

where `m` is the number of available depth sources.

So the old source is not deleted. It only must overcome the penalty through
content relevance.

## 3. New Failure Mode

The penalty creates a tradeoff:

```text
c too small:
  stale milestone clutter remains competitive.

c too large:
  old but necessary assumptions become unrecoverable.
```

Therefore `c` is not a theorem knob to set by taste. It must be validated by
two tests:

```text
recent-step preference test
old-assumption recall test
```

## 4. Relation To Milestone Gate

Distance penalty should not replace time-axis preservation. It only controls
readout priority among available summaries.

```text
Milestone gate preserves the state.
Distance-penalized AttnRes chooses which preserved summary to emphasize.
```

If the milestone gate fails to preserve a critical state, the distance penalty
cannot recover it.

## 5. Implementation Contract

The first implementation should expose:

```text
attnres_distance_penalty_c
distance definition: depth_distance | source_age | milestone_age
logs of beta distribution entropy
logs of old-critical-source attention mass
ablation with c = 0
```

This keeps the extension falsifiable.
