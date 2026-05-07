# Formal Proof B3: Gate Stability Under Utility Estimation Error

Created: 2026-05-03

Status: robustness proof for B1/B2 when `Delta_i` is estimated rather than known.

Correction status: optional B-time proof for estimated utilities in sparse
anchor/cache gates. It is not required by the active Dense RetNet + Block
AttnRes + Engram baseline.

## 0. Why B3 Is Needed

B1 and B2 assume exact oracle utilities:

```text
Delta_i = L_without_anchor_i - L_with_anchor_i
```

In practice, the gate can only use estimates:

```text
Delta_hat_i
```

B3 gives deterministic error conditions under which the selected gate remains
unchanged.

## 1. Setup

Assume a uniform utility-estimation error bound:

```text
|Delta_hat_i - Delta_i| <= eta
```

for all candidate anchors `i`.

## 2. Theorem B3.1: Threshold Gate Is Stable Under A 2-Eta Margin

**Statement**

For a fixed threshold `lambda`, suppose every token has margin at least `eta`
from the threshold in the true utility scale:

```text
|Delta_i - lambda| > eta
```

Then the threshold decision using estimates matches the oracle decision:

```text
1[Delta_hat_i > lambda] = 1[Delta_i > lambda]
```

for every `i`.

**Proof**

If `Delta_i > lambda`, then by the margin assumption:

```text
Delta_i - lambda > eta
```

Using the error bound:

```text
Delta_hat_i >= Delta_i - eta > lambda
```

so both oracle and estimated gates select `i`.

If `Delta_i < lambda`, then:

```text
lambda - Delta_i > eta
```

and:

```text
Delta_hat_i <= Delta_i + eta < lambda
```

so both oracle and estimated gates reject `i`.

QED.

## 3. Theorem B3.2: Top-B Gate Is Stable Under A 2-Eta Boundary Gap

Sort true utilities:

```text
Delta_(1) >= Delta_(2) >= ... >= Delta_(n)
```

Assume:

```text
Delta_(B) > 0
Delta_(B) - Delta_(B+1) > 2 eta
```

Then selecting the top `B` utilities by `Delta_hat_i` recovers the same top-`B`
set as selecting by true `Delta_i`.

**Proof**

Let `i` be any true top-`B` item and `j` be any item outside the true top `B`.
Then:

```text
Delta_i >= Delta_(B)
Delta_j <= Delta_(B+1)
```

Using the estimation bound:

```text
Delta_hat_i >= Delta_i - eta >= Delta_(B) - eta
Delta_hat_j <= Delta_j + eta <= Delta_(B+1) + eta
```

The boundary gap assumption gives:

```text
Delta_(B) - eta > Delta_(B+1) + eta
```

Therefore:

```text
Delta_hat_i > Delta_hat_j
```

for every true top-`B` item `i` and every non-top-`B` item `j`. The estimated
top-`B` set must equal the true top-`B` set.

QED.

## 4. Consequence For Training

The sparse gate does not need perfect utilities. It needs utility estimates whose
error is smaller than the relevant selection margin:

```text
eta < (Delta_(B) - Delta_(B+1)) / 2
```

This is stricter than merely having low average error. A few high-error estimates
near the boundary can change the selected cache set.

## 5. Relation To A And C

A and C affect utility estimation:

- A affects whether an anchor truly changes future loss through non-decaying
  paths.
- C affects whether Engram-injected information changes loss or only adds noise.

B3 says that even if useful anchors exist, the learned gate is stable only when
their estimated utility margin is wider than estimation error.

Therefore a rigorous experimental plan must report:

```text
estimated Delta_i
empirical Delta_i from ablation
estimation error eta
boundary gap Delta_(B) - Delta_(B+1)
cache selection stability under repeated seeds
```
