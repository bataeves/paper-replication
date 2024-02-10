from typing import Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from papers.vit.blocks.patch_position import PatchPositionBlock
from papers.vit.blocks.transformer import TransformerEncoderBlock
from papers.vit.const import IMG_SIZE, PATCH_SIZE
from papers.vit.layers.patch_embedding import PatchEmbedding


class ViTModel(LightningModule):
    def __init__(
        self,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_transformer_layers: int = 12,
        num_heads: int = 12,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        color_channels: int = 3,
        num_classes: int = 1000,
        loss: torch.nn.Module | None = None,
        learning_rate: float = 3e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.3,
    ):
        """
        Vision Transformer Model

        Args:
            num_transformer_layers: How many Transformer Encoder blocks are there? (each of these will contain a MSA block and MLP block)
            embedding_dim: This is the embedding dimension throughout the architecture, this will be the size of the
                vector that our image gets turned into when it gets patched and embedded.
            mlp_size: What are the number of hidden units in the MLP layers?
            num_heads: How many heads are there in the Multi-Head Attention layers?
            img_size: Original image size in (Height, Width) format
            patch_size: Patch size
            color_channels: Number of color channels in input image
            attn_dropout: Dropout for attention projection
            mlp_dropout: Dropout for dense/MLP layers
            embedding_dropout: Dropout for patch and position embeddings
            num_classes: Default for ImageNet but can customize this
        """
        super().__init__()

        assert (
            img_size % patch_size == 0
        ), f"Image size must be divisible by patch size. {img_size=}, {patch_size=}."

        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay

        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.img_size = img_size
        self.patch_size = patch_size
        self.color_channels = color_channels

        self.layer_patch_position = PatchPositionBlock(
            PatchEmbedding(
                patch_size=patch_size,
                in_channels=color_channels,
                embedding_dim=embedding_dim,
            ),
            embedding_dropout=embedding_dropout,
        )
        transformers = [
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_transformer_layers)
        ]
        self.transformer_encoder = nn.Sequential(*transformers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )
        self.loss = loss or torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
        # Create patch+position embedding
        x = self.layer_patch_position(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0])

        return x

    def _calc_loss(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calc_loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._calc_loss(batch))

    def test_step(self, batch, batch_idx):
        self.log("test_loss", self._calc_loss(batch))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
