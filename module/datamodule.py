import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/cifar10",
        dataset="cifar10",
        batch_size: int = 128,
        normalization=True,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if normalization:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            self.transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.transform_test = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.cifar_train = CIFAR10(root=self.data_dir, train=True, transform=self.transform_train)
        self.cifar_test = CIFAR10(root=self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
