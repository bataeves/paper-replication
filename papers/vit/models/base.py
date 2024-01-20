from typing import Tuple

from papers.vit.const import IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE


class VitBase:
    def __init__(
        self,
        layers: int = 12,
        hidden_size: int = 768,
        mlp_size: int = 3072,
        heads: int = 12,
        img_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
        patch_size: int = PATCH_SIZE,
        color_channels: int = 3,
    ):
        """
        Vision Transformer Model

        Args:
            layers: How many Transformer Encoder blocks are there? (each of these will contain a MSA block and MLP block)
            hidden_size: This is the embedding dimension throughout the architecture, this will be the size of the
                vector that our image gets turned into when it gets patched and embedded.
            mlp_size: What are the number of hidden units in the MLP layers?
            heads: How many heads are there in the Multi-Head Attention layers?
            img_size: Original image size in (Height, Width) format
            patch_size: Patch size
            color_channels: Number of color channels in input image
        """
        self.layers = layers
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size
        self.heads = heads
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.color_channels = color_channels

    @property
    def num_patches(self) -> int:
        """
        Returns: The resulting number of patches, which also serves as the input sequence length for the Transformer.
        """
        return int((self.img_height * self.img_width) / (self.patch_size**2))

    @property
    def embedding_layer_input_shape(self) -> Tuple[int, int, int]:
        return self.img_height, self.img_width, self.color_channels

    @property
    def embedding_layer_output_shape(self) -> Tuple[int, int]:
        return self.num_patches, self.patch_size**2 * self.color_channels
