from torch import nn


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0
    ):
        """

        Args:
            embedding_dim: Hidden size D from Table 1 for ViT-Base
            num_heads: Heads from Table 1 for ViT-Base
            attn_dropout: doesn't look like the paper uses any dropout in MSABlocks
        """
        super().__init__()
        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # does our batch dimension come first?
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,  # do we need the weights or just the layer outputs?
        )
        return attn_output
