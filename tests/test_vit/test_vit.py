import pytest
import torch

from papers.vit.models.vit_base import ViTBase


# Values based on https://www.learnpytorch.io/08_pytorch_paper_replicating/#327-exploring-table-1
@pytest.mark.parametrize(
    "img_size,patch_size,layers,embedding_dim,mlp_size,heads",
    [
        (224, 16, 12, 768, 3072, 12),
        (224, 16, 24, 1024, 4096, 16),
        (224, 16, 32, 1280, 5120, 16),
    ],
)
def test_vit_base(img_size, patch_size, layers, embedding_dim, mlp_size, heads):
    color_channels = 3
    num_classes = 30
    model = ViTBase(
        img_size=img_size,
        patch_size=patch_size,
        num_transformer_layers=layers,
        mlp_size=mlp_size,
        num_heads=heads,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        color_channels=color_channels,
    )
    input_shape = (1, color_channels, img_size, img_size)
    x = torch.rand(input_shape)
    output = model(x)
    assert output.shape == (1, num_classes)
