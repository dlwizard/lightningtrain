from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.vision import StandardTransform
from torchvision import transforms as T
from typing import Optional
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: ImageFolder,
        test_dataset: ImageFolder,
        image_size: int = 32,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        train_dataset.transform = self.transforms
        test_dataset.transform = self.transforms

        self.data_train = train_dataset
        self.data_val = test_dataset
        self.data_test = test_dataset

    @property
    def num_classes(self):
        return len(self.data_train.classes)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )