import torch

from papers.vit.blocks.msa import MultiheadSelfAttentionBlock


def test_msa():
    shape = (1, 197, 768)
    x = torch.rand(shape)

    block = MultiheadSelfAttentionBlock(embedding_dim=768, num_heads=12)
    output = block(x)
    assert output.shape == shape
