<div align="center">

# Engram Retention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20041183.svg)](https://doi.org/10.5281/zenodo.20041183)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](LICENSE)
[![CI](https://github.com/XXY-CH/engram-retention/actions/workflows/ci.yml/badge.svg)](https://github.com/XXY-CH/engram-retention/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](requirements.txt)

**A proof-aligned PyTorch scaffold for budgeted long-context memory.**

</div>

Reference PyTorch code and proof notes for a Dense-first long-context language
modeling research scaffold that combines:

- RetNet-style recurrent retention for compact sequence mixing.
- Block Attention Residuals for depth-wise reuse of intermediate states.
- Hashed Engram lookup for deterministic N-gram residual memory.
- Optional milestone-triggered gates and bounded snapshot readout for selected
  token-time facts.

The current claim is deliberately bounded: this repository studies auxiliary
memory paths that may improve sparse long-context recall under explicit budget
constraints. It does not claim a universal replacement for full KV-cache
attention.

## Status

This is an early research codebase. The implementation is intended for
architecture experiments, synthetic diagnostics, and proof-aligned ablations.
Interfaces may change while the core assumptions are refined.

## Architecture

Engram Retention is organized as a small language-modeling stack plus a proof
trail. The implementation keeps four mechanisms separate so each one can be
ablated and audited:

```text
input tokens
    |
    v
token + position embeddings
    |
    v
Dense RetNet-Engram layers
    |
    |-- RetentionLayer
    |     compact recurrent/parallel sequence mixing with fixed per-head decay
    |
    |-- Dense FFN
    |     phase-1 channel mixing baseline; MoE remains a future extension
    |
    |-- HashedNgramEngram
    |     deterministic N-gram lookup injected as a gated residual branch
    |
    |-- BlockAttentionResidual
    |     depth-axis reuse over earlier layer/block states
    |
    |-- MilestoneRetentionGate + MilestoneSnapshotReadout
    |     optional selected-token preservation and bounded snapshot readout
    |
    v
RMSNorm + tied output projection
    |
    v
next-token logits and diagnostic metrics
```

The research stack around the model is:

```text
src/            implementation surface
tests/          behavior locks for layers, model wiring, and training smoke paths
experiments/    synthetic tasks, ablations, and reproducibility configs
analysis/       plotting scripts for curated figures
docs/           methodology, proof trail, assumption audits, and progress notes
papers/         human-written notes over local paper readings
references/     BibTeX and paper manifest, without committing mirrored PDFs
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a lighter environment without packaging metadata:

```bash
python -m pip install -r requirements.txt
```

## Quick Checks

```bash
python -m pytest
python experiments/train_synthetic.py --task needle --variants ours,retnet,transformer --steps 5 --batch-size 4 --seq-len 32 --log-interval 5 --out-dir experiments/results/smoke
```

## Repository Layout

```text
analysis/                   Figure-generation scripts and derived analysis
docs/                       Research methodology, architecture notes, proofs, progress
experiments/                Synthetic diagnostics and experiment configs
papers/                     Reading notes for RetNet, Engram, and attention residuals
references/                 BibTeX and manifest for the external paper corpus
src/                        PyTorch implementation
tests/                      Unit and integration smoke tests
```

PDF papers, local virtual environments, generated plots, temporary extraction
text, and agent runtime state are intentionally excluded from Git.

## Example Synthetic Run

```bash
python experiments/train_synthetic.py \
  --task needle \
  --variants ours_snapshot_logits,retnet,transformer \
  --use-milestones \
  --use-snapshots \
  --use-snapshot-logit-bias \
  --steps 200 \
  --eval-batches 4 \
  --out-dir experiments/results/needle_snapshot_eval
```

The runner writes CSV metrics and a loss plot under the selected output
directory. Generated results are ignored by default; commit only curated
summaries or figures that are part of a paper artifact.

## Research Trail

Start with:

- `docs/proofs/proof_plan.md`
- `docs/proofs/pdf_assumption_audit_2026-05-03.md`
- `docs/proofs/25-general-llm-replacement-necessary-conditions.md`
- `docs/progress/2026-05-04-external-review-response.md`

The proof trail separates RetNet recurrence, depth-axis residual reuse,
Engram-style lookup, and milestone snapshots so that ablations can be interpreted
without overstating the architecture's scope.

## Citation

If this repository helps your work, cite it through `CITATION.cff` or the
Zenodo DOI: `10.5281/zenodo.20041183`.

## License

Creative Commons Attribution 4.0 International (`CC-BY-4.0`). See `LICENSE`.
