# Engram Notes

## Biological Background

**Engram** - The physical trace of memory in the brain, a persisting change in neural tissue following learning.

### Key Theories

1. **Complementary Learning Systems (CLS)** - McClelland et al. (1995)
   - Hippocampus: fast learning, episodic memory
   - Neocortex: slow consolidation, semantic memory

2. **Synaptic Consolidation** - Memory traces stabilize over time
   - Early: labile, plastic
   - Late: consolidated, resistant to interference

3. **Memory Reactivation** - Replay strengthens engrams
   - Sharp-wave ripples in hippocampus
   - Sleep consolidation

### Computational Analogues

- **Experience Replay** - DQN, prioritized experience replay
- **Memory-Augmented Neural Networks** - NTM, DNC
- **Fast Weights** - Hinton's fast weights, linear transformers
- **Persistent Memory** - Transformer memory tokens, Memorizing Transformers

### Mapping to RetNet

| Biological Concept | Computational Analog |
|---|---|
| Engram formation | High-attention state persistence |
| Synaptic consolidation | Memory gate hardening over steps |
| Memory reactivation | Selective retrieval from retention buffer |
| Interference | Catastrophic forgetting mitigation |

### Research Questions

- Can engram gates selectively persist high-value states in RetNet?
- Does a consolidation schedule improve long-context performance?
- How does engram-based memory compare to explicit KV-cache?

## References

- Tonegawa et al. (2015) - Memory Engram Cells Have Come of Age
- McClelland et al. (1995) - Why There Are Complementary Learning Systems
- Frankland & Bontempi (2005) - The Organization of Recent and Remote Memories

## Experimental Notes

_Add experimental observations here_
