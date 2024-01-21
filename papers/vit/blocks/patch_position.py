import torch
from torch import nn

from papers.vit.layers.patch_embedding import PatchEmbedding


class PatchPositionBlock(nn.Module):
    def __init__(self, patch_embedding: PatchEmbedding, embedding_dropout: float = 0.1):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x):
        patch_embedding = self.patch_embedding(x)
        batch_size, number_of_patches, embedding_dimension = patch_embedding.shape

        class_token = nn.Parameter(
            torch.rand(batch_size, 1, embedding_dimension), requires_grad=True
        )

        # Prepend class token embedding to patch embedding
        patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)

        # Create position embedding
        position_embedding = nn.Parameter(
            torch.rand(1, number_of_patches + 1, embedding_dimension), requires_grad=True
        )

        # Add position embedding to patch embedding with class token
        patch_and_position_embedding = patch_embedding_class_token + position_embedding
        return self.dropout(patch_and_position_embedding)
