import lightning as L
import torch


class MulticlassLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def _calc_loss(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calc_loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._calc_loss(batch))

    def test_step(self, batch, batch_idx):
        self.log("test_loss", self._calc_loss(batch))

    def configure_optimizers(self):
        return self.optimizer
