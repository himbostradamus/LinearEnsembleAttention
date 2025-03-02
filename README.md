# Linear Ensemble Attention
Linear Ensemble Transformer (LET): A transformer architecture that enhances attention through learnable linear combinations of multiple attention maps with tanh-weighted coefficients. Generalizes DIFF Transformer by allowing both positive and negative attention map contributions.

Introduction
Linear Ensemble Transformer (LET) extends traditional transformer architectures by introducing a learnable linear combination of multiple attention maps, enhancing the model's ability to focus on relevant information while actively canceling noise.
LET generalizes the Differential Transformer (DIFF Transformer) approach by:

1. Supporting any number of attention maps (not just two)
2. Using tanh normalization to enable both positive and negative weights (-1 to +1)
3. Learning optimal combinations of attention patterns through training

## Acknowledgments

This work builds upon and generalizes the Differential Transformer approach introduced in ["DIFFERENTIAL TRANSFORMER" (Ye et al., 2024)](https://arxiv.org/abs/2410.05258) ([code repository](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)).
