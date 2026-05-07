# Layers

This folder contains reusable neural network components.

## Files

- `retention.py` - RetNet-style parallel and recurrent retention layer.
- `engram.py` - Deterministic hashed N-gram Engram residual branch.
- `attention_residual.py` - Block Attention Residual depth-reuse branch.
- `milestone_gate.py` - Optional milestone-conditioned retention gate.
- `milestone_snapshot.py` - Bounded snapshot collection and readout.

Layer modules should stay small, typed, and independently testable.
