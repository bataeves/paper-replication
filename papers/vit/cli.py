import lightning as L
import torch
import typer

from papers.const import DATA_CACHE_DIR
from papers.utils.data import download_data, create_dataloaders
from papers.vit.data import get_transforms
from papers.vit.models.vit_base import ViTBase
from papers.vit.trainer import MulticlassLightningModule

app = typer.Typer()


@app.command()
def train_pss(
    img_size: int = 224,
    epochs: int = 7,
    batch_size: int = 32,
):
    """
    Train pizza, steak, sushi model
    """
    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination=DATA_CACHE_DIR / "pizza_steak_sushi",
    )
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=image_path / "train",
        test_dir=image_path / "test",
        transform=get_transforms(img_width=img_size, img_height=img_size),
        batch_size=batch_size,
    )
    model = ViTBase(
        num_classes=len(class_names),
    )
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.999),
        weight_decay=0.3,
    )
    lmodule = MulticlassLightningModule(
        model=model,
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(),
    )
    trainer = L.Trainer(
        max_epochs=epochs,
    )
    trainer.fit(lmodule, train_dataloader)
    trainer.test(lmodule, test_dataloader)


@app.command()
def demo():
    pass


if __name__ == "__main__":
    app()
