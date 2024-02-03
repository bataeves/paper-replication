import lightning as L
import torch
import typer

from papers.tasks.image.classification import ImageClassificationTask
from papers.vit.data import get_transforms
from papers.vit.models.vit_base import ViTBase
from papers.vit.trainer import MulticlassLightningModule

app = typer.Typer(
    help="Vision transformer implementation based on PyTorch Paper Replicating article.",
    short_help="Vision transformer implementation",
)


@app.command()
def train(
    dataset: str,
    img_size: int = 224,
    epochs: int = 7,
    batch_size: int = 32,
):
    """
    Train pizza, steak, sushi model
    """
    task = ImageClassificationTask.from_code(dataset)
    class_names = task.class_names

    dl_kwargs = {
        "transform": get_transforms(img_width=img_size, img_height=img_size),
        "batch_size": batch_size,
    }

    train_dataloader = task.dataloader_train(**dl_kwargs)
    test_dataloader = task.dataloader_test(**dl_kwargs)

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
    trainer = L.Trainer(max_epochs=epochs)
    trainer.fit(lmodule, train_dataloader)
    trainer.test(lmodule, test_dataloader)


@app.command()
def demo():
    pass


if __name__ == "__main__":
    app()
