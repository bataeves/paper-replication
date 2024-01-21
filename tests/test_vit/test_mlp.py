import torch

from papers.vit.blocks.mlp import MultilayerPerceptronBlock


def test_mlp():
    embedding_dim = 768
    mlp_size = 3072
    shape = (1, 197, embedding_dim)
    x = torch.rand(shape)

    block = MultilayerPerceptronBlock(
        embedding_dim=embedding_dim,
        mlp_size=mlp_size,
    )
    output = block(x)
    assert output.shape == shape
