# Paper Manifest

Created: 2026-05-03

This manifest records the local paper corpus for the Dense RetNet + Engram +
Block Attention Residuals research program. Keep this file as the
source-of-truth for what has actually been fetched into the workspace.

## Core Papers

| Topic | Paper | Local file | Source URL | Use in proof |
|---|---|---|---|---|
| RetNet backbone | Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models" | `references/papers/core/retnet_2307.08621.pdf` | https://arxiv.org/abs/2307.08621 | Defines retention, recurrent representation, parallel/recurrent/chunkwise modes, and the O(1) per-token inference claim for the backbone. |
| Conditional memory / Engram | Cheng et al., "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models" | `references/papers/core/engram_conditional_memory_2601.07372.pdf` | https://arxiv.org/abs/2601.07372 | Defines Engram N-gram hashing, context-aware gating, residual integration, topology-agnostic adaptation, offload/prefetch, and reported reasoning/long-context gains. |
| Discrete multimodal tokens | van den Oord et al., "Neural Discrete Representation Learning" | `references/papers/core/vq_vae_1711.00937.pdf` | https://arxiv.org/abs/1711.00937 | Supports the claim that continuous modalities can be represented through discrete latent codes before Engram hashing. |

## Related Papers

| Topic | Paper | Local file | Source URL | Use in proof |
|---|---|---|---|---|
| Transformer baseline | Vaswani et al., "Attention Is All You Need" | `references/papers/related/attention_is_all_you_need_1706.03762.pdf` | https://arxiv.org/abs/1706.03762 | Baseline for full attention and residual stream formulation. |
| Linear attention recurrence | Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" | `references/papers/related/linear_attention_2006.16236.pdf` | https://arxiv.org/abs/2006.16236 | Background for recurrent/linear attention equivalences and state update analysis. |
| Deep residual scaling | Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" | `references/papers/related/deepnet_2203.00555.pdf` | https://arxiv.org/abs/2203.00555 | Background for residual scaling and gradient stability in deep attention networks. |
| External memory | Wu et al., "Memorizing Transformers" | `references/papers/related/memorizing_transformers_2203.08913.pdf` | https://arxiv.org/abs/2203.08913 | Reference point for non-differentiable key-value memory and approximate retrieval. |
| Linear-MoE | Sun et al., "Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts" | `references/papers/related/linear_moe_2503.05447.pdf` | https://arxiv.org/abs/2503.05447 | Archived future-extension evidence only. MoE is intentionally excluded from the active Dense baseline. |
| Feature hashing | Weinberger et al., "Feature Hashing for Large Scale Multitask Learning" | `references/papers/related/feature_hashing_0902.2206.pdf` | https://arxiv.org/abs/0902.2206 | Mathematical support for signed hashing, collision effects, and hash-space dimensionality tradeoffs in Engram analysis. |
| Hash embeddings | Svenstrup et al., "Hash Embeddings for Efficient Word Representations" | `references/papers/related/hash_embeddings_1709.03933.pdf` | https://arxiv.org/abs/1709.03933 | Direct ancestor for multi-hash embedding tables used by Engram-style N-gram memory. |
| Product-key memory | Lample et al., "Large Memory Layers with Product Keys" | `references/papers/related/product_key_memory_1907.05242.pdf` | https://arxiv.org/abs/1907.05242 | Reference for large neural memory layers with fast lookup and large capacity. |
| Infini-gram | Liu et al., "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens" | `references/papers/related/infini_gram_2401.17377.pdf` | https://arxiv.org/abs/2401.17377 | Modern large-scale N-gram system evidence for lookup-style language modeling. |
| Model editing | Meng et al., "Locating and Editing Factual Associations in GPT" | `references/papers/related/rome_2202.05262.pdf` | https://arxiv.org/abs/2202.05262 | Background for factual association editing; relevant to but not proof of Engram slot editing. |
| Mass model editing | Meng et al., "Mass-Editing Memory in a Transformer" | `references/papers/related/memit_2210.07229.pdf` | https://arxiv.org/abs/2210.07229 | Background for large-scale memory editing; useful for evaluating online knowledge update claims. |
| Parametric vs non-parametric memory | Mallen et al., "When Not to Trust Language Models..." | `references/papers/related/nonparametric_memory_trust_2212.10511.pdf` | https://arxiv.org/abs/2212.10511 | Helps frame when external/non-parametric memory is preferable to parametric recall. |
| Count-min sketch | Cormode and Muthukrishnan, "An Improved Data Stream Summary: The Count-Min Sketch and its Applications" | `references/papers/related/count_min_sketch_2005.pdf` | https://www.cs.helsinki.fi/u/jilu/paper/countMin.pdf | Reference for multi-hash sketch capacity/error reasoning. |
| Tokenization | Kudo and Richardson, "SentencePiece..." | `references/papers/related/sentencepiece_1808.06226.pdf` | https://arxiv.org/abs/1808.06226 | Background for tokenizer behavior; relevant to Engram tokenizer-compression / canonicalization. |
| Normalization | Zhang and Sennrich, "Root Mean Square Layer Normalization" | `references/papers/related/rmsnorm_1910.07467.pdf` | https://arxiv.org/abs/1910.07467 | Supports RMSNorm/gating normalization details used in Engram fusion. |
| Biological memory systems | McClelland, McNaughton, O'Reilly, "Why There Are Complementary Learning Systems..." | `references/papers/related/complementary_learning_systems_1995.pdf` | https://imss-www.upmf-grenoble.fr/prevert/MasterIC2A/SpecialiteSC/FichiersPDF/Why%20there%20are%20complementary%20learning%20systems%20in%20the%20hippocampus%20and%20neocortex%20insights%20from%20th.pdf | Supports the fast-learning / slow-consolidation analogy behind Engram-style external memory. |

## Pre-Existing Local Papers Corpus

The project root also contains a pre-existing `papers/` corpus. On this
filesystem, `papers/` and `Papers/` point to the same directory entries for the
inspected subdirectories; use lowercase paths as canonical in notes.

| Topic | Paper / file | Local file | Use in proof |
|---|---|---|---|
| Attention residuals | Kimi Team, "Attention Residuals" | `papers/attention-residual/Attention Residuals.pdf` | Direct support for depth-wise attention over preceding layer/block outputs; useful residual-aggregation primitive, but not a direct proof of token-time sparse CoT anchors. |
| Residual learning | He et al., "Deep Residual Learning for Image Recognition" | `papers/attention-residual/He2016_Deep_Residual_Learning.pdf` | Foundation for identity shortcut/residual gradient-flow framing. |
| Dense cross-layer connections | Huang et al., "Densely Connected Convolutional Networks" | `papers/attention-residual/Huang2017_Densely_Connected_CNNs.pdf` | Background for cross-layer feature reuse. |
| Highway gates | Srivastava et al., "Highway Networks" | `papers/attention-residual/Srivastava2015_Highway_Networks.pdf` | Background for gated residual/carry paths. |
| LayerScale | Touvron et al., "Going deeper with Image Transformers" | `papers/attention-residual/Touvron2021_LayerScale_Residual_Scaling.pdf` | Supports learnable residual scaling for deep Transformers. |
| Engram duplicate / source copy | Cheng et al., "Conditional Memory via Scalable Lookup" | `papers/engram/Engram2026_Memory_Consolidation.pdf` | Same Engram paper family as the core corpus; local copy used to verify architecture details and section evidence. |
| RetNet duplicate / source copy | Sun et al., "Retentive Network" | `papers/retnet/Sun2023_Retentive_Network.pdf` | Same RetNet paper as core corpus. |

## Hardware References

| Topic | Document | Local file | Source URL | Use in proof |
|---|---|---|---|---|
| Orange Pi 4 Pro / A733 | Orange Pi 4 Pro A733 User Manual v1.4 | `references/papers/hardware/orangepi_4_pro_user_manual_v1.4.pdf` | https://orangepi.net/wp-content/uploads/2026/01/OrangePi_4_Pro_A733_User-Manual_v1.4.pdf | Grounding for edge hardware constraints: A733 SoC, 3 TOPS NPU, memory/storage interfaces. |
| A733 product page | Allwinner A733 product page | Not mirrored yet | https://www.allwinnertech.com/index.php?c=product&id=139 | Independent vendor grounding for 3 TOPS NPU and LPDDR4/LPDDR4x/LPDDR5 + UFS/eMMC support. |

## Missing Or Not Yet Mirrored

| Topic | Target | Status | Next action |
|---|---|---|---|
| Biological Engram review | Tonegawa et al., "Memory Engram Cells Have Come of Age" | Metadata verified through PubMed, local PDF mirror attempted but failed with HTTP 404. | Use PubMed/ScienceDirect metadata for citation; find a stable open PDF mirror if direct text is needed. |
| Complementary learning systems | McClelland, McNaughton, O'Reilly, 1995 | Mirrored from a university-hosted PDF after initial manifest creation. | Prefer a publisher DOI/source citation in final bibliography if available. |

## Retrieval Notes

- Fetched with `curl -L --fail` on 2026-05-03.
- `pdftotext` is not installed in this environment; detailed formula extraction should either use a PDF reader manually or install/use a PDF parsing tool in a later step.
- The RetNet file fetched from arXiv is compact because arXiv serves the short paper PDF for `2307.08621`.
