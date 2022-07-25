import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from modules.utils_transfer import get_mobilenetv2


def get_accuracy(logits: torch.Tensor, y: torch.Tensor):
    return ((logits > 0.0) == y).float().mean()


class CatsDogsClassifier(pl.LightningModule):
    def __init__(self, imagenet_weights=True, dropout=0.5, lr=1e-4):
        super().__init__()
        self.classifier = get_mobilenetv2(
            num_class=1, pretrained_weights=imagenet_weights, dropout=dropout
        )
        self.lr = lr

    def forward(self, x):
        logits = self.classifier(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def __compute_batch_loss(self, batch):
        x, y, _ = batch
        y = y.unsqueeze(axis=1)
        x = self.classifier(x)
        loss = F.binary_cross_entropy_with_logits(x, y, reduction="mean")
        batch_size = len(x)
        return loss, get_accuracy(x, y), batch_size

    def training_step(self, train_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(train_batch)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_accuracy", accuracy, batch_size=batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(val_batch)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_accuracy", accuracy, batch_size=batch_size)

    def test_step(self, test_batch, batch_idx):
        loss, accuracy, batch_size = self.__compute_batch_loss(test_batch)
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_accuracy", accuracy, batch_size=batch_size)
