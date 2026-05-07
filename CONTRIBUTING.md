# Contributing

Thanks for your interest in improving Engram Retention.

This repository is research-first, so contributions should keep code changes
small, reproducible, and tied to an explicit experiment or proof claim.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull Request Expectations

- Keep behavior changes covered by tests or a synthetic diagnostic command.
- Document new research assumptions under `docs/proofs/` or `docs/progress/`.
- Do not commit generated experiment directories, model checkpoints, PDF mirrors,
  local agent state, or private notes.
- Prefer small ablations over broad architecture rewrites.
- Separate Dense-first phase-1 evidence from future MoE or scaling extensions.

## Code Style

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

The public API is still experimental. If you are changing interfaces in
`src/layers/` or `src/models/`, include a short rationale in the PR description.
