import torch

from papers.vit.blocks.transformer import TransformerEncoderBlock


def test_transformer():
    shape = (1, 197, 768)
    x = torch.rand(shape)
    block = TransformerEncoderBlock()

    output = block(x)
    assert output.shape == shape
