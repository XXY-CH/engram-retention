# Engram Retention

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
src/layers/                 Core retention, Engram, AttnRes, and snapshot layers
src/models/                 RetNet-Engram model and Transformer baseline
src/training/               Small training utilities
experiments/train_synthetic.py
                            Synthetic diagnostic runner
experiments/configs/        Reproducible experiment configurations
tests/                      Unit and integration smoke tests
docs/proofs/                Proof trail and assumption audits
docs/progress/              Research progress notes
references/bibtex/          Bibliography for the paper/proof stack
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
