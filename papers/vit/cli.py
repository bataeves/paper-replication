from typing import Annotated, Tuple

import lightning as pl
import typer
import wandb

from papers.integrations.wandb import init, get_pl_logger
from papers.tasks.image.classification import ImageClassificationTask
from papers.vit.data import get_transforms
from papers.vit.model import ViTModel

app = typer.Typer(
    help="Vision transformer implementation based on PyTorch Paper Replicating article.",
    short_help="Vision transformer implementation",
)


@app.command()
def train(
    dataset: str,
    img_size: Annotated[
        int, typer.Option(help="Preprocessing image size", rich_help_panel="Data loading")
    ] = 224,
    epochs: Annotated[
        int, typer.Option(help="Number of training epochs", rich_help_panel="Training")
    ] = 7,
    learning_rate: Annotated[
        float, typer.Option(help="Learning rate", rich_help_panel="Training")
    ] = 3e-3,
    betas: Annotated[
        Tuple[float, float], typer.Option(help="Optimizer Beta", rich_help_panel="Training")
    ] = (0.9, 0.999),
    weight_decay: Annotated[
        float,
        typer.Option(
            help="Weight decay",
            rich_help_panel="Training",
        ),
    ] = 0.3,
):
    init(group=dataset)

    task = ImageClassificationTask.get_task_class(dataset)
    data_module = task(
        transform=get_transforms(img_width=img_size, img_height=img_size),
    )
    wandb.config["data"] = data_module.get_params()

    model = ViTModel(
        num_classes=len(data_module.class_names),
        img_size=img_size,
        learning_rate=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        logger=get_pl_logger(),
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    app()
