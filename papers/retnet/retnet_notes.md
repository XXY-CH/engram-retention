# RetNet Paper Notes

## Core Paper

**Retentive Network: A Successor to Transformer for Large Language Models**
- Authors: Sun et al. (2023)
- Key innovation: Retention mechanism with three computation paradigms

### Key Concepts

1. **Parallel Retention** - Training-parallel computation
2. **Recurrent Retention** - Inference-step recurrence (O(1) memory)
3. **Chunkwise Retention** - Balanced approach for long sequences

### Mathematical Formulation

```
Retention(X) = (Q @ K^T * D) @ V
where D is a causal decay mask incorporating relative position
```

### Open Questions
- How does retention scale with depth?
- Can retention be made more expressive with residual connections?
- Memory consolidation across retention steps?

## Related Papers

_Add notes from related RetNet papers here_
