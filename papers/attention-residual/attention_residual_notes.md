# Attention Residual Notes

## Core Concepts

### Residual Connections in Attention

Standard Transformer:
```
x = x + Attention(x)
x = x + FFN(x)
```

### Variants to Investigate

1. **Dense Attention** - Skip connections across multiple attention layers
2. **Highway Attention** - Gated residual in attention computation
3. **Attention over Residuals** - Attention applied to residual stream directly
4. **DeepNet-style scaling** - Sub-LayerNorm with depth-dependent scaling

### Key Papers

- He et al. (2016) - Deep Residual Learning
- Huang et al. (2017) - DenseNet
- Srivastava et al. (2015) - Highway Networks
- Wang et al. (2022) - DeepNet: Scaling Transformers to 1,000 Layers
- Zhang & Zuo (2023) - LayerScale and residual scaling

### Research Questions

- Can attention residual improve RetNet depth scaling?
- What is the optimal residual topology for retention layers?
- How do engram gates interact with residual connections?

## Experimental Notes

_Add experimental observations here_
