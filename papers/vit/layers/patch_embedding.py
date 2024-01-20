from torch import nn


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self, patch_size: int, embedding_dim: int, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        """A layer to turn an image into patches"""

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        """A layer to flatten the patch feature maps into a single dimension"""

    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        if image_resolution % self.patch_size != 0:
            raise ValueError(
                f"Input image size must be divisble by patch size. "
                f"Image shape: {image_resolution}, patch size: {self.patch_size}"
            )

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # Make sure the output shape has the right order
        # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]s
        return x_flattened.permute(0, 2, 1)
