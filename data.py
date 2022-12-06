import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np


class MRData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].astype("float32"), self.y[idx].astype("float32")


class MRDataModule(pl.LightningDataModule):
    def __init__(
        self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, num_workers=0
    ):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = MRData(self.x_train, self.y_train)
        self.val_dataset = MRData(self.x_val, self.y_val)
        self.test_dataset = MRData(self.x_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
