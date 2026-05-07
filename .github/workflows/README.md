# GitHub Actions Workflows

CI lives here.

## Current Workflow

- `ci.yml` installs the project in editable mode with development dependencies
  and runs `pytest` on Python 3.10 and 3.11.

The workflow is intentionally small: it proves that the public package metadata
and tests work from a clean checkout without requiring local paper PDFs,
generated experiment results, or private runtime state.
