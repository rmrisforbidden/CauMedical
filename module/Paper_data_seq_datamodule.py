import pwd
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import os
import scipy.io
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from script.trainer import BlochSeqEqDecoder, BlochImgEqDecoder
from Bloch_decoder.utils.generate_RF_TR import generate_RF_TR


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 1, num_workers=0, test_type="seq", subsamp=5, seq_jump=5, is_input_RF=0,
    ):  # img_cnn, img_seq, seq
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_type = test_type
        self.subsamp = subsamp
        self.seq_jump = seq_jump
        self.datamodule = ApproxSeqDataset_seperate_RF if is_input_RF>0 else PingSeqDataset_seperate

    def prepare_data(self):
        return

    def setup(self, stage=None):
        if "img" in self.test_type:
            self.trainset = self.datamodule(mode='train')
            self.testset = imgSeqDataset(subsamp=self.subsamp) #if "seq" in self.type else imgCnnDataset(subsamp=self.subsamp)
        else:
            self.trainset = self.datamodule(mode='train')
            self.testset = self.datamodule(mode='test')
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if "img" in self.test_type:
             return DataLoader(self.testset, batch_size=128*128, shuffle=False, num_workers=self.num_workers)
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class ApproxSeqDataset_seperate_RF(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(self, mode='train', is_T1T2_logscale=False, is_TETR_microsecond=True):
        """
        Approx RNN input : (is_T1T2_logscale=True, is_TETR_microsecond=False)
        Our Bloch decoder input : (is_T1T2_logscale=False, is_TETR_microsecond=True)
        This data : (is_T1T2_logscale=True, is_TETR_microsecond=False)
        """

        # Load data from paper

        # if mode == 'train':
        #     MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start1_Train.mat')
        # else:
        #     MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start5_Val.mat')
        # self.D = torch.from_numpy(np.real(MRFData['D'][:,1::5]))
        # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)
        # self.labels = torch.from_numpy(MRFData['LUT']) 

        # T1T2 scale
        self.T1 = self.labels[:,0].unsqueeze(1).repeat(1, L//subsamp)
        self.T2 = self.labels[:,1].unsqueeze(1).repeat(1, L//subsamp)
        if is_T1T2_logscale == False:
            self.T1 = torch.pow(10, self.T1)
            self.T2 = torch.pow(10, self.T2)

        # RF, TR, TE
        L=1000
        subsamp=5
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        self.RFpulses = RFpulses.repeat(len(self.labels),1)
        self.TR = TR[0:L:subsamp].repeat(len(self.labels),1)
        self.TE = torch.ones(len(self.labels), L//subsamp)*10

        if is_TETR_microsecond == True:
            self.TR = self.TR*1000
            self.TE = self.TE*1000
        
        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR),0)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class PingSeqDataset_mixed(Dataset):
    def __init__(self):
        MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start1_Train.mat') 
        MRFData_Val = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start5_Val.mat') #
        self.labels = np.concatenate((MRFData['LUT'], MRFData_Val['LUT']),0)
        self.D = np.concatenate((np.real(MRFData['D']), np.real(MRFData_Val['D'])),0)
        self.D = self.D[:,1::5]
        
        self.D = torch.from_numpy(self.D)
        self.labels = torch.from_numpy(self.labels)
        self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class PingSeqDataset_seperate(Dataset):
    def __init__(self, mode='train'):

        if mode == 'train':
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start1_Train.mat')
        else:
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/D_LUT_L1000_TE10_Start5_Val.mat')
        self.D = torch.from_numpy(np.real(MRFData['D'][:,1::5]))
        self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)
        self.labels = torch.from_numpy(MRFData['LUT']) 

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


