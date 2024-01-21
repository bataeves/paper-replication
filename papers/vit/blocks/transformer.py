"""
Transformer Encoder - The Transformer Encoder, is a collection of the layers listed above.
There are two skip connections inside the Transformer encoder (the "+" symbols) meaning the layers's inputs are fed
directly to immediate layers as well as subsequent layers. The overall ViT architecture is comprised of a number of
Transformer encoders stacked on top of eachother.
"""
from torch import nn

from papers.vit.blocks.mlp import MultilayerPerceptronBlock
from papers.vit.blocks.msa import MultiheadSelfAttentionBlock


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0,
    ):
        """

        Args:
            embedding_dim: Hidden size D from Table 1 for ViT-Base
            num_heads: Heads from Table 1 for ViT-Base
            mlp_size: MLP size from Table 1 for ViT-Base
            mlp_dropout: Amount of dropout for dense layers from Table 3 for ViT-Base
            attn_dropout: Amount of dropout for attention layers
        """
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        self.mlp_block = MultilayerPerceptronBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        # Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x
