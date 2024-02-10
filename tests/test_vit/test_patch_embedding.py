from papers.vit.layers.patch_embedding import PatchEmbedding


def test_patch(dataloader):
    patch_size = 16
    embedding_dim = 768
    patchify = PatchEmbedding(
        in_channels=3,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
    )
    image_batch, label_batch = next(iter(dataloader))
    image, label = image_batch[0], label_batch[0]

    patch_input = image.unsqueeze(0)

    patch_embedded_image = patchify(patch_input)

    assert patch_embedded_image.shape == (1, 196, embedding_dim)
