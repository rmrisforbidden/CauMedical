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
        self, batch_size: int = 1, num_workers=0, test_type="seq", subsamp=1, seq_jump=5, is_input_RF=0, need_T1T2_logscale=False, need_TETR_second=False,need_RF_degree=False,
    ):  # img_cnn, img_seq, seq
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_type = test_type
        self.subsamp = subsamp
        self.seq_jump = seq_jump
        self.need_T1T2_logscale = need_T1T2_logscale
        self.need_TETR_second = need_TETR_second
        self.need_RF_degree = need_RF_degree
        self.datamodule = PingSeqDataset_seperate_RF if is_input_RF>0 else PingSeqDataset_seperate

    def prepare_data(self):
        return

    def setup(self, stage=None):
        if "img" in self.test_type:
            self.trainset = self.datamodule(mode='train', need_T1T2_logscale=self.need_T1T2_logscale, need_TETR_second=self.need_TETR_second, need_RF_degree=self.need_RF_degree)
            self.testset = PingImgDataset_seperate_RF(mode='test', need_T1T2_logscale=self.need_T1T2_logscale, need_TETR_second=self.need_TETR_second, need_RF_degree=self.need_RF_degree)
        else:
            self.trainset = self.datamodule(mode='train', need_T1T2_logscale=self.need_T1T2_logscale, need_TETR_second=self.need_TETR_second, need_RF_degree=self.need_RF_degree)
            self.testset = self.datamodule(mode='test', need_T1T2_logscale=self.need_T1T2_logscale, need_TETR_second=self.need_TETR_second, need_RF_degree=self.need_RF_degree)
            
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


class PingSeqDataset_seperate_RF(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(self, mode='train', need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """
        if mode == 'train':
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start1_Train.mat')
        else:
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start5_Val.mat')
        L=1000
        subsamp=1
        self.D = torch.from_numpy(np.real(MRFData['D'][:,0:L:subsamp]))
        self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)
        self.labels = torch.from_numpy(MRFData['LUT']) 
        

        # T1T2 scale
        len_seq = L//subsamp
        self.T1 = self.labels[:,0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)
        self.T2 = self.labels[:,1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)
        if need_T1T2_logscale == True:
            self.T1 = torch.log10(self.T1)
            self.T2 = torch.log10(self.T2)

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        self.RFpulses = RFpulses.repeat(len(self.labels),1).unsqueeze(1)
        self.TR = TR[0:L:subsamp].repeat(len(self.labels),1).unsqueeze(1)
        self.TE = torch.ones(len(self.labels), 1, len_seq)*10

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR/1000
            self.TE = self.TE/1000
                
        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR),1).transpose(1,2) # (80100, 5, 200)
        
        # Simpler
        # self.labels = self.labels[0:20]
        # self.D = self.D[0:20]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class PingImgDataset_seperate_RF(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(self, mode='test', need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """
        MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/MRI_N128_L1000_TE10.mat')
        L=1000
        subsamp=1
        self.D = torch.from_numpy(np.real(MRFData['X_fullysamp'][:,0:L:subsamp])) # self.D:(128,128,1000)
        self.D = torch.flatten(self.D, 0, 1) # self.D:(128*128,1000)
        self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        # T1T2 scale
        len_seq = L//subsamp
        self.T1 = torch.flatten(torch.from_numpy(MRFData['T1_128']), 0, 1) # self.T1:(128*128,)
        self.T2 = torch.flatten(torch.from_numpy(MRFData['T2_128']), 0, 1) # self.T2:(128*128,)
        self.T1 = self.T1.unsqueeze(1).repeat(1, len_seq).unsqueeze(1) # self.T1:(128*128,lenth, 1)
        self.T2 = self.T2.unsqueeze(1).repeat(1, len_seq).unsqueeze(1) # self.T2:(128*128,lenth, 1)
        if need_T1T2_logscale == True:
            self.T1 = torch.log10(self.T1)
            self.T2 = torch.log10(self.T2)

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        self.RFpulses = RFpulses.repeat(len(self.T1),1).unsqueeze(1)
        self.TR = TR[0:L:subsamp].repeat(len(self.T1),1).unsqueeze(1)
        self.TE = torch.ones(len(self.T1), 1, len_seq)*10

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR/1000
            self.TE = self.TE/1000
                
        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR),1).transpose(1,2) # (128*128, 5, 1000)
        
        # Simpler
        # self.labels = self.labels[0:20]
        # self.D = self.D[0:20]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class PingSeqDataset_mixed(Dataset):
    def __init__(self):
        MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start1_Train.mat') 
        MRFData_Val = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start5_Val.mat') #
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
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start1_Train.mat')
        else:
            MRFData = scipy.io.loadmat('lightning_bolts/Bloch_decoder/data/Pingfan/D_LUT_L1000_TE10_Start5_Val.mat')
        self.D = torch.from_numpy(np.real(MRFData['D'][:,1::5]))
        self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)
        self.labels = torch.from_numpy(MRFData['LUT']) 

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class SeqDataset(Dataset):
    def __init__(self, subsamp=5, seq_jump=5):
        self.data_file_name = (
            "lightning_bolts/Bloch_decoder/data/generated_sequence_"
            + str(seq_jump)
            + "_normalized_subsamp"
            + str(subsamp)
            + "_under800.npz"
        )
        print(self.data_file_name)
        self.subsamp = subsamp
        self.seq_jump = seq_jump

        if not os.path.isfile(self.data_file_name):
            self.D, self.labels = self.generate_data()
            self.D = self.D.detach().cpu()
            self.labels = self.labels.detach().cpu()
        else:
            self.D = torch.from_numpy(np.load(self.data_file_name)["x"])
            self.labels = torch.from_numpy(np.load(self.data_file_name)["y"])

    def generate_data(
        self,
    ):
        # Generate labels
        T1_values = torch.range(0, 5000, self.seq_jump).cuda()  # torch.range(10, 20, 5).cuda()  #
        T2_values = torch.range(0, 100, self.seq_jump).cuda()  # torch.range(0, 10, 5).cuda()  #
        T1 = T1_values.repeat_interleave(len(T2_values))
        T2 = T2_values.repeat(len(T1_values))
        labels = torch.stack((T1, T2), -1).cuda()

        # Generate sequence
        bloch = BlochSeqEqDecoder(subsamp=self.subsamp)
        D = bloch.forward(labels).detach()
        D = D.type(torch.float)
        print(D)
        np.savez(self.data_file_name, x=D.detach().cpu(), y=labels.detach().cpu())
        return D, labels

    def min_max_T(self, T1_values, T2_values):
        n_T1 = len(T1_values)
        n_T2 = len(T2_values)
        max_T = T1_values if n_T1 >= n_T2 else T2_values
        min_T = T1_values if n_T1 < n_T2 else T2_values
        min_T = min_T.repeat((len(max_T) // len(min_T)) + 1)[: len(max_T)]
        [self.T1_values, self.T2_values] = [max_T, min_T] if n_T1 >= n_T2 else [min_T, max_T]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class imgSeqDataset(Dataset):
    def __init__(self, subsamp=5):
        self.data_file_name = ("lightning_bolts/Bloch_decoder/data/MRI_Size128x128_L1000_Subsamp" + str(subsamp) + ".npz")
        self.subsamp = subsamp

        if not os.path.isfile(self.data_file_name):
            self.D, self.labels = self.generate_data()
            self.D = self.D.detach().cpu()
            self.labels = self.labels.detach().cpu()

        else:
            self.D = torch.from_numpy(np.load(self.data_file_name)["MRIs"])
            T12Maps = np.load(self.data_file_name, allow_pickle=True)["T12Maps"].item()
            T1_image = torch.from_numpy(T12Maps["T1_128"])  # .type(torch.float)
            T2_image = torch.from_numpy(T12Maps["T2_128"])  # .type(torch.float)
            # PD_image = torch.from_numpy(T12Maps["PD_128"])  # .type(torch.float)
            self.labels = torch.stack((T1_image, T2_image), -1)
            w, h = self.labels.shape[0], self.labels.shape[1]
            self.labels = self.labels.reshape(w * h, -1)

        # Treat (128*128) as the number of data
        print("---------------- data_size ---------------- ")
        print(self.D.shape, self.labels.shape)
        print("-" * 50)

    def generate_data(self):
        T12_data_path = "lightning_bolts/Bloch_decoder/data/Groundtruth_T1_T2_PD.mat"
        T12Maps = scipy.io.loadmat(T12_data_path)
        # print(T12Maps.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'T2_128', 'T1_128', 'PD_128'])
        T1_image = T12Maps["T1_128"]
        T2_image = T12Maps["T2_128"]
        # PD_image = T12Maps["PD_128"]

        T1_image = torch.from_numpy(T1_image)
        T2_image = torch.from_numpy(T2_image)
        # PD_image = torch.from_numpy(PD_image)

        labels = torch.stack((T1_image, T2_image), -1).cuda()
        w, h = labels.shape[0], labels.shape[1]
        labels = labels.reshape(w * h, -1)

        bloch = BlochSeqEqDecoder(subsamp=self.subsamp)
        D = bloch.forward(labels).detach()
        D = D.type(torch.float)
        np.savez(self.data_file_name, MRIs=D.detach().cpu().numpy(), T12Maps=T12Maps)
        return D, labels

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]

class imgCnnDataset(Dataset):
    def __init__(self):
        self.data_file_name = "lightning_bolts/Bloch_decoder/data/MRI_Size128x128_L1000_Subsamp20.npz"
        # assert os.path.isfile(self.data_file_name), "You shoud generate data"

        if not os.path.isfile(self.data_file_name):
            self.D, self.labels = self.generate_data()
            self.D = self.D.transpose(1, 2).transpose(0, 1).detach().cpu()
            self.labels = self.labels.transpose(0, 1).transpose(1, 2).detach().cpu()

            # Repeat
            self.D = torch.stack((self.D, self.D, self.D, self.D, self.D, self.D), 0)
            self.labels = torch.stack(
                (self.labels, self.labels, self.labels, self.labels, self.labels, self.labels), 0
            )

        else:
            self.D = torch.from_numpy(np.load(self.data_file_name)["MRIs"]).type(torch.float)
            T12Maps = np.load(self.data_file_name, allow_pickle=True)["T12Maps"].item()
            T1_image = torch.from_numpy(T12Maps["T1_128"])  # .type(torch.float)
            T2_image = torch.from_numpy(T12Maps["T2_128"])  # .type(torch.float)
            PD_image = torch.from_numpy(T12Maps["PD_128"])  # .type(torch.float)
            self.labels = torch.stack((T1_image, T2_image, PD_image), -1)

            # Treat (128*128) as the number of data
            # w, h = self.D.shape[0], self.D.shape[1]
            # self.D = self.D.reshape(w * h, -1)
            # self.labels = self.labels.reshape(w * h, -1)

            # Treat (1,) as the number of data
            self.D = self.D.transpose(2, 1).transpose(1, 0)
            self.D = self.D.unsqueeze(0)
            self.labels = self.labels.unsqueeze(0)

            # Repeat
            self.D = self.D.squeeze(0)
            self.labels = self.labels.squeeze(0)
            self.D = torch.stack((self.D, self.D), 0)
            self.labels = torch.stack((self.labels, self.labels), 0)
            print(self.D.shape, self.labels.shape)
            print("99" * 30)

    def generate_data(self):
        T12_data_path = "lightning_bolts/Bloch_decoder/data/Groundtruth_T1_T2_PD.mat"
        T12Maps = scipy.io.loadmat(T12_data_path)
        # print(T12Maps.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'T2_128', 'T1_128', 'PD_128'])
        T1_image = T12Maps["T1_128"]
        T2_image = T12Maps["T2_128"]
        PD_image = T12Maps["PD_128"]

        T1_image = torch.from_numpy(T1_image)
        T2_image = torch.from_numpy(T2_image)
        PD_image = torch.from_numpy(PD_image)

        labels = torch.stack((T1_image, T2_image, PD_image), -1)

        bloch = BlochImgEqDecoder(subsamp=20)
        D = bloch.forward(labels).detach()
        D = D.type(torch.float)
        print(D.shape)
        print("000000" * 30)
        np.savez(self.data_file_name, MRIs=D.detach().cpu().numpy(), T12Maps=T12Maps)
        return D, labels

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]
