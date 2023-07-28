from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy

class GPTLitModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate=1e-3,
        block_size=8,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ignoring net as the model weights themselves are not a hyperparam
        self.save_hyperparameters(logger=False, ignore=['model'])

        self.learning_rate = self.hparams.learning_rate

        self.model = model

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.register_buffer("mask", torch.tril(torch.ones(self.hparams.block_size, self.hparams.block_size)) == 0)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        mask = self.mask if targets is not None else None
        return self.model(x, targets=targets, mask=mask)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, targets=y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate)

        return {"optimizer": optimizer}