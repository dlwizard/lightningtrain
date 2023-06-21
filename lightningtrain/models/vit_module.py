from typing import Any

from lightning import LightningModule

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
    ):
        super(PatchEmbedding, self).__init__()

        assert (
            img_size / patch_size % 1 == 0
        ), "img_size must be integer multiple of patch_size"

        self.projection = nn.Sequential(
            Rearrange(
                "b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=patch_size, s2=patch_size
            ),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.positional_emb = nn.Parameter(
            torch.randn(
                (img_size // patch_size) * (img_size // patch_size)
                + 1,  # 14 x 14 patches + CLS patch
                emb_size,
            )
        )

    def forward(self, x):
        B, *_ = x.shape
        x = self.projection(x)
        # print(x.shape, )
        cls_token = repeat(self.cls_token, "() p e -> b p e", b=B)

        # print(cls_token.shape)

        x = torch.cat([cls_token, x], dim=1)

        x += self.positional_emb

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.projection = nn.Linear(emb_size, emb_size)

        self.attn_dropout = nn.Dropout(dropout)

        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x, mask=None):
        rearrange_heads = (
            "batch seq_len (num_head h_dim) -> batch num_head seq_len h_dim"
        )

        queries = rearrange(self.query(x), rearrange_heads, num_head=self.num_heads)
        keys = rearrange(self.key(x), rearrange_heads, num_head=self.num_heads)

        values = rearrange(self.key(x), rearrange_heads, num_head=self.num_heads)

        energies = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(energies.dtype).min
            energies.mask_fill(~mask, fill_value)

        attention = F.softmax(energies, dim=-1) * self.scaling

        attention = self.attn_dropout(attention)

        out = torch.einsum("bhas, bhsd -> bhad", attention, values)

        out = rearrange(
            out, "batch num_head seq_length dim -> batch seq_length (num_head dim)"
        )

        out = self.projection(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x

        out = self.fn(x, **kwargs)

        out += res

        return out

FeedForwardBlock = lambda emb_size=768, expansion=4, drop_p=0.0: nn.Sequential(
    nn.Linear(emb_size, expansion * emb_size),
    nn.GELU(),
    nn.Dropout(drop_p),
    nn.Linear(expansion * emb_size, emb_size),
)

class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size=768, drop_p=0.0, forward_expansion=4, forward_drop_p=0, **kwargs
    ):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super(TransformerEncoder, self).__init__(
            *(TransformerEncoderBlock(**kwargs) for _ in range(depth))
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, num_classes=1000):
        super(ClassificationHead, self).__init__(
            Reduce(
                "batch_size seq_len emb_dim -> batch_size emb_dim", reduction="mean"
            ),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes),
        )

class ViT(nn.Sequential):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
        depth=12,
        num_classes=1000,
        **kwargs
    ):
        super(ViT, self).__init__(
            PatchEmbedding(
                in_channels,
                patch_size,
                emb_size,
                img_size,
            ),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes),
        )

class VitLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes=10,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = ViT(
            in_channels=3,
            patch_size=4,
            emb_size=64,
            img_size=32,
            depth=6,
            num_classes=num_classes,
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            <https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers>
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}