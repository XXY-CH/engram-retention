# Source Code

This directory contains the PyTorch implementation.

## Contents

- `layers/` - Reusable architecture primitives.
- `models/` - Complete language model assemblies and baselines.
- `training/` - Small training utilities used by tests and diagnostics.
- `utils/` - Reserved for shared utility code.

The source layout follows the proof decomposition: keep RetNet recurrence,
Engram lookup, Block AttnRes, and milestone snapshots separable so ablations
remain interpretable.
