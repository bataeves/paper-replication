"""
MLP (or Multilayer perceptron) - A MLP can often refer to any collection of feedforward layers (or in PyTorch's case,
a collection of layers with a forward() method). In the ViT Paper, the authors refer to the MLP as "MLP blocks" and it
contains two torch.nn.Linear() layers with a torch.nn.GELU() non-linearity activation in between them (section 3.1) and
a torch.nn.Dropout() layers after each (Appendex B.1).
"""
from torch import nn


class MultilayerPerceptronBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    def __init__(self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1):
        """

        Args:
            embedding_dim: Hidden Size D from Table 1 for ViT-Base
            mlp_size: MLP size from Table 1 for ViT-Base
            dropout: Dropout from Table 3 for ViT-Base
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=mlp_size,
                out_features=embedding_dim,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
