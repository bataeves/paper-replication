from papers.vit.blocks.patch_position import PatchPositionBlock
from papers.vit.layers.patch_embedding import PatchEmbedding


def test_patch_position(dataloader):
    patch_position = PatchPositionBlock(
        PatchEmbedding(
            in_channels=3,
            patch_size=16,
            embedding_dim=768,
        )
    )

    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    patch_input = image.unsqueeze(0)

    output = patch_position(patch_input)

    assert output.shape == (1, 197, 768)
