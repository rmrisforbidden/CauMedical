import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

import brainweb
import matplotlib.pyplot as plt

import brainweb
from brainweb import volshow
import numpy as np
from os import path
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)


class BrainWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/brain",
        dataset="brainweb",
        batch_size: int = 128,
        normalization=True,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        files = brainweb.get_files()

        # stack shape : (20, 254, 144, 144)
        brainweb.seed(1337)
        stack = None

        for f in tqdm(files, desc="mMR ground truths", unit="subject"):
            vol = brainweb.get_mmr_fromfile(
                f,
                petNoise=1,
                t1Noise=0.75,
                t2Noise=0.75,
                petSigma=1,
                t1Sigma=1,
                t2Sigma=1,
            )
            # Cut out background
            t1 = vol["T1"][:, 100:-100, 100:-100]
            t2 = vol["T2"][:, 100:-100, 100:-100]

            if stack is None:
                stack = np.concatenate((np.expand_dims(t1, 0), np.expand_dims(t2, 0)), axis=1)

            else:
                stack = np.concatenate((stack, np.concatenate((np.expand_dims(t1, 0), np.expand_dims(t2, 0)), axis=1)))

        self.stack = torch.from_numpy(stack).to(dtype=torch.float)

    def setup(self, stage=None):
        self.braintrain = TensorDataset(
            self.stack[:15],
            torch.ones(self.stack[:15].shape[0]),
        )
        self.braintest = TensorDataset(
            self.stack[15:],
            torch.ones(self.stack[15:].shape[0]),
        )

    def train_dataloader(self):
        return DataLoader(self.braintrain, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.braintest, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.braintest, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.braintest, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
