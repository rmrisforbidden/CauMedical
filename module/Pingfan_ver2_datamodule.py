import pwd

from jinja2 import ModuleLoader
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import os
import scipy.io
import sys
import mat73
import h5py
from pytorch_lightning.trainer.supporters import CombinedLoader
import scipy.io as sio

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from script.trainer import BlochSeqEqDecoder, BlochImgEqDecoder
from Bloch_decoder.utils.generate_RF_TR import generate_RF_TR


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers=0,
        train_data_type="seq",
        test_data_type="real",
        subsamp=1,
        seq_jump=5,
        is_input_RF=0,
        need_T1T2_logscale=True,
        need_TETR_second=True,
        need_RF_degree=True,
    ):  # img_cnn, img_seq, seq
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_type = train_data_type
        self.testdata_type = test_data_type
        self.subsamp = subsamp
        self.seq_jump = seq_jump
        self.need_T1T2_logscale = need_T1T2_logscale
        self.need_TETR_second = need_TETR_second
        self.need_RF_degree = need_RF_degree
        self.datamodule = PingSeqDataset

        self.n_img_tr = 80 * 128 * 128
        self.n_img_te = 22 * 128 * 128

    def prepare_data(self):
        return

    def setup(self, stage=None):
        # Trainset
        if "RF" in self.train_data_type:
            self.datamodule = PingSeqDataset_mixedRF
            is_split_range_T1T2 = False if "seqAll" in self.train_data_type else True
            print("is_split_range_T1T2 : ", is_split_range_T1T2, "-------------------------")
            self.trainset = self.datamodule(
                mode="train",
                is_split_range_T1T2=is_split_range_T1T2,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
                type=self.train_data_type,
            )
        elif "seqTr" in self.train_data_type:
            self.trainset = self.datamodule(
                mode="train",
                is_split_range_T1T2=True,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
        elif "seqAll" in self.train_data_type:
            self.trainset = self.datamodule(
                mode="none",
                is_split_range_T1T2=False,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
        elif "phantomTr" in self.train_data_type:
            self.trainset = PingImgDataset_Phantom(mode="train")
        elif "phantomAll" in self.train_data_type:
            self.trainset = PingImgDataset_Phantom(mode="none")
        else:
            raise AssertionError("Check traindata_type")

        # Testset
        self.test_seqTe = self.datamodule(
            mode="test",
            is_split_range_T1T2=True,
            need_T1T2_logscale=self.need_T1T2_logscale,
            need_TETR_second=self.need_TETR_second,
            need_RF_degree=self.need_RF_degree,
            type=self.train_data_type,
        )
        # self.test_seqTe_RF = PingSeqDataset_RF(
        #     mode="test",
        #     is_split_range_T1T2=False,
        #     need_T1T2_logscale=self.need_T1T2_logscale,
        #     need_TETR_second=self.need_TETR_second,
        #     need_RF_degree=self.need_RF_degree,
        # )
        # self.test_phantomTe = PingImgDataset_Phantom(mode="test")
        # self.test_phantomAll_RF = PingImgDataset_Phantom_RF(mode="none")
        self.test_phantomAll = PingImgDataset_Phantom(mode="none")
        self.test_real = PingImgDataset_real(
            need_T1T2_logscale=self.need_T1T2_logscale,
            need_TETR_second=self.need_TETR_second,
            need_RF_degree=self.need_RF_degree,
        )
        # self.test_real_noise = PingImgDataset_real_noise(
        #     need_T1T2_logscale=self.need_T1T2_logscale,
        #     need_TETR_second=self.need_TETR_second,
        #     need_RF_degree=self.need_RF_degree,
        #     type=self.train_data_type,
        # )

        # if "seqTe" in self.testdata_type:
        #     self.testset = self.datamodule(
        #         mode="test",
        #         is_split_range_T1T2=True,
        #         need_T1T2_logscale=self.need_T1T2_logscale,
        #         need_TETR_second=self.need_TETR_second,
        #         need_RF_degree=self.need_RF_degree,
        #     )
        # elif "phantomTe" in self.testdata_type:
        #     self.testset = PingImgDataset_Phantom(mode="test")
        # elif "phantomAll" in self.testdata_type:
        #     self.testset = PingImgDataset_Phantom(mode="none")
        # elif "real" in self.testdata_type:
        #     self.testset = PingImgDataset_real(
        #         need_T1T2_logscale=self.need_T1T2_logscale,
        #         need_TETR_second=self.need_TETR_second,
        #         need_RF_degree=self.need_RF_degree,
        #     )
        # else:
        #     raise AssertionError("Check testdata_type")

    def setup_old(self, stage=None):

        if "dictionary" in self.data_type:
            # seq train / seq test
            # seq_all train / real or phantom_all test
            # phantom train / phantom test
            # phantom all / real test

            # Trainset
            if "train_seqTR" in self.data_type:
                self.trainset = self.datamodule(
                    mode="train",
                    is_split_range_T1T2=True,
                    need_T1T2_logscale=self.need_T1T2_logscale,
                    need_TETR_second=self.need_TETR_second,
                    need_RF_degree=self.need_RF_degree,
                )
            elif "train_seqAll" in self.data_type:
                self.trainset = self.datamodule(
                    mode="none",
                    is_split_range_T1T2=False,
                    need_T1T2_logscale=self.need_T1T2_logscale,
                    need_TETR_second=self.need_TETR_second,
                    need_RF_degree=self.need_RF_degree,
                )
            elif "train_phantomTr" in self.data_type:
                self.trainset = PingImgDataset_Phantom(mode="train")
            elif "train_phantomAll" in self.data_type:
                self.trainset = PingImgDataset_Phantom(mode="none")

            # Test set
            if "test_seqTE" in self.data_type:
                self.testset = self.datamodule(
                    mode="test",
                    is_split_range_T1T2=True,
                    need_T1T2_logscale=self.need_T1T2_logscale,
                    need_TETR_second=self.need_TETR_second,
                    need_RF_degree=self.need_RF_degree,
                )
            elif "test_phantomTe" in self.data_type:
                self.testset = PingImgDataset_Phantom(mode="test")
            elif "test_phantomAll" in self.data_type:
                self.testset = PingImgDataset_Phantom(mode="none")
            elif "test_real" in self.data_type:
                self.testset = PingImgDataset_real(
                    need_T1T2_logscale=self.need_T1T2_logscale,
                    need_TETR_second=self.need_TETR_second,
                    need_RF_degree=self.need_RF_degree,
                )

        elif "img_img" in self.data_type:
            self.trainset = PingImgDataset_Phantom(mode="train")
            self.testset = PingImgDataset_Phantom(mode="test")

        elif "seq_img" in self.data_type:
            self.trainset = self.datamodule(
                mode="train",
                is_split_range_T1T2=True,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
            self.testset = PingImgDataset_Phantom(mode="test")
        elif "seq_seq" in self.data_type:
            self.trainset = self.datamodule(
                mode="train",
                is_split_range_T1T2=True,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
            self.testset = self.datamodule(
                mode="test",
                is_split_range_T1T2=True,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
        elif "img_none" in self.data_type:  # This is for training RNN
            self.trainset = PingImgDataset_Phantom(mode="none")
            self.testset = PingImgDataset_Phantom(mode="none")

        elif "seq_none" in self.data_type:  # This is for training RNN
            self.trainset = self.datamodule(
                mode="none",
                is_split_range_T1T2=False,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
            self.testset = self.datamodule(
                mode="none",
                is_split_range_T1T2=False,
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
        elif "real_img" in self.data_type:
            self.trainset = PingImgDataset_real(
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
            self.testset = PingImgDataset_real(
                need_T1T2_logscale=self.need_T1T2_logscale,
                need_TETR_second=self.need_TETR_second,
                need_RF_degree=self.need_RF_degree,
            )
        else:
            raise AssertionError("data_type should be in [img_img, seq_seq, seq_img, img_none, seq_none]")

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):

        # Validation daata loader for all data types
        seqTe = DataLoader(self.test_seqTe, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # seqTe_RF = DataLoader(
        #     self.test_seqTe_RF, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        # )
        # phantomTe = DataLoader(
        #     self.test_phantomTe, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        # )
        phantomAll = DataLoader(
            self.test_phantomAll, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        # phantomAll_noise = DataLoader(
        #     self.test_phantomAll_RF, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        # )
        real = DataLoader(self.test_real, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # real_noise = DataLoader(
        #     self.test_real_noise, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        # )

        loaders = {
            "seqTe": seqTe,
            "real": real,
            "phantomAll": phantomAll,
            # "phantomAll_noise": phantomAll_noise,
            # "seqTe_RF": seqTe_RF,
            # "real_noise": real_noise,
        }  # "phantomTe": phantomTe, "phantomAll": phantomAll, "real": real}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class PingSeqDataset(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(
        self,
        mode="train",
        is_split_range_T1T2=True,
        need_T1T2_logscale=False,
        need_TETR_second=False,
        need_RF_degree=False,
        type=None,
    ):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """

        L = 1000
        subsamp = 1
        len_seq = L // subsamp
        T1_condition_threshold = 2500
        T2_condition_threshold = 1000
        MRFData = scipy.io.loadmat(
            "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/New_D_LUT_L1000_TE10_Start1_Train.mat"
        )
        MRFData_Val = scipy.io.loadmat(
            "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/New_D_LUT_L1000_TE10_Start5_Val.mat"
        )  #
        self.is_split_range_T1T2 = is_split_range_T1T2 if "none" not in mode else False

        self.labels = np.concatenate((MRFData["LUT"], MRFData_Val["LUT"]), 0)
        self.labels = torch.from_numpy(self.labels)

        self.D = np.concatenate((MRFData["D"], MRFData_Val["D"]), 0)
        self.D = torch.from_numpy(self.D[:, 0:L:subsamp])

        # Already normalized in New... files
        # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        if self.is_split_range_T1T2:
            # Split train and test set : T1 0~2500, 2500~5000
            T1 = self.labels[:, 0]
            T2 = self.labels[:, 1]
            condition = (
                (T1 < T1_condition_threshold) & (T2 < T2_condition_threshold)
                if "train" in mode
                else (T1 > T1_condition_threshold) & (T2 > T2_condition_threshold)
            )
            self.labels = self.labels[condition]
            self.D = self.D[condition]

        # T1T2 scale
        self.T1 = self.labels[:, 0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)
        self.T2 = self.labels[:, 1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)

        if need_T1T2_logscale == True:

            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(self.T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(self.T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            self.D = self.D[T1T2_idx_n100]
            self.T1 = T1_log[T1T2_idx_n100]
            self.T2 = T2_log[T1T2_idx_n100]

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.

        self.RFpulses = RFpulses.repeat(len(self.labels), 1).unsqueeze(1)
        self.TR = TR[0:L:subsamp].repeat(len(self.labels), 1).unsqueeze(1)
        self.TE = torch.ones(len(self.labels), 1, len_seq) * 10

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR / 1000
            self.TE = self.TE / 1000

        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
            1, 2
        )  # (80100, 5, 200)

        # This if for Debug
        # self.labels = self.labels[0:20]
        # self.D = self.D[0:20]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingSeqDataset_RF(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(
        self,
        mode="train",
        is_split_range_T1T2=True,
        need_T1T2_logscale=False,
        need_TETR_second=False,
        need_RF_degree=False,
        type="Spline5",
    ):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """

        L = 1000
        subsamp = 1
        len_seq = L // subsamp
        T1_condition_threshold = 2500
        T2_condition_threshold = 1000
        MRFData = scipy.io.loadmat(
            "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/New_D_LUT_L1000_TE10_Start1_Train.mat"
        )
        # "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_1Spline5_D_LUT_L1000_TE10_Start1_Train.mat"
        MRFData_Val = scipy.io.loadmat(
            "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/New_D_LUT_L1000_TE10_Start5_Val.mat"
        )  # num100_5PieceConstant5Flex_D_LUT_L1000_TE10_Start5_Val
        self.is_split_range_T1T2 = is_split_range_T1T2 if "none" not in mode else False

        self.labels = np.concatenate((MRFData["LUT"], MRFData_Val["LUT"]), 0)  # MRFData["LUT"]  #
        self.D = np.concatenate((MRFData["D"], MRFData_Val["D"]), 0)  # MRFData["D"]  #

        self.labels = torch.from_numpy(self.labels)
        self.D = torch.from_numpy(self.D[:, 0:L:subsamp])

        # Already normalized in New... files
        # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.

        self.TR = TR[0:L:subsamp].repeat(len(self.labels), 1).unsqueeze(1)
        self.TE = torch.ones(len(self.labels), 1, len_seq) * 10
        self.RFpulses = RFpulses.repeat(len(self.labels), 1).unsqueeze(1)
        ###########################
        # Add noise on RF
        # name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"
        # name2 = "1Spline5Flex"  #'2Spline11'#'2Spline11Flex' '5PieceConstant5Flex' '4SplineNoise11Flex' '1Spline5Flex'
        # file_name = name + name2 + ".mat"
        # traindata = sio.loadmat(file_name)
        # RFpulses = torch.tensor(traindata["rf"][100, :1000])
        # RFpulses = RFpulses * torch.pi / 180
        # RFpulses = RFpulses.repeat(len(self.labels), 1).unsqueeze(1)

        # name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"
        # name2 = "5PieceConstant5Flex"  #'2Spline11'#'2Spline11Flex' '5PieceConstant5Flex' '4SplineNoise11Flex' '1Spline5Flex'
        # file_name = name + name2 + ".mat"
        # traindata = sio.loadmat(file_name)
        # RFpulses2 = torch.tensor(traindata["rf"][100, :1000])
        # RFpulses2 = RFpulses2 * torch.pi / 180
        # RFpulses2 = RFpulses2.repeat(len(self.labels), 1).unsqueeze(1)

        # # Concat
        # self.RFpulses = torch.cat((RFpulses[: len(MRFData["LUT"])], RFpulses2[len(MRFData["LUT"]) :]), 0)
        ###########################

        if self.is_split_range_T1T2:
            # Split train and test set : T1 0~2500, 2500~5000
            T1 = self.labels[:, 0]
            T2 = self.labels[:, 1]
            condition = (
                (T1 < T1_condition_threshold) & (T2 < T2_condition_threshold)
                if "train" in mode
                else (T1 > T1_condition_threshold) & (T2 > T2_condition_threshold)
            )
            self.labels = self.labels[condition]
            self.D = self.D[condition]
            self.RFpulses = self.RFpulses[condition]
            self.TE = self.TE[condition]
            self.TR = self.TR[condition]

        # T1T2 scale
        self.T1 = self.labels[:, 0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)
        self.T2 = self.labels[:, 1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)

        if need_T1T2_logscale == True:

            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(self.T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(self.T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            self.D = self.D[T1T2_idx_n100]
            self.T1 = T1_log[T1T2_idx_n100]
            self.T2 = T2_log[T1T2_idx_n100]
            self.RFpulses = self.RFpulses[T1T2_idx_n100]
            self.TE = self.TE[T1T2_idx_n100]
            self.TR = self.TR[T1T2_idx_n100]

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR / 1000
            self.TE = self.TE / 1000

        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
            1, 2
        )  # (80100, 5, 200)

        # This if for Debug
        # self.labels = self.labels[0:20]
        # self.D = self.D[0:20]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingSeqDataset_mixedRF(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(
        self,
        mode="train",
        is_split_range_T1T2=True,
        need_T1T2_logscale=False,
        need_TETR_second=False,
        need_RF_degree=False,
        type=None,
    ):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """

        L = 1000
        subsamp = 1
        len_seq = L // subsamp
        T1_condition_threshold = 2500
        T2_condition_threshold = 1000

        data_list = []
        if "005" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/0.05noise_D_LUT_L1000_TE10_Start5_Val.mat"
            )  # num100_5PieceConstant5Flex_D_LUT_L1000_TE10_Start5_Val
            data_list.append(MRFData)
        if "03" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/0.3noise_D_LUT_L1000_TE10_Start5_Val.mat"
            )  # num100_5PieceConstant5Flex_D_LUT_L1000_TE10_Start5_Val
            data_list.append(MRFData)
        if "01" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/0.1noise_D_LUT_L1000_TE10_Start5_Val.mat"
            )  #
            data_list.append(MRFData)
        if "1Spline5Flex" in type and "num250" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num250_1Spline5Flex_D_LUT_L1000_TE10_Start5_Val.mat"
            )  #
            data_list.append(MRFData)

        if "1Spline5Flex" in type and "num200" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_1Spline5Flex_D_LUT_L1000_TE10_Start5_Val.mat"
            )  #
            data_list.append(MRFData)

        if "4SplineNoise11Flex" in type and "num200" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_4SplineNoise11Flex_D_LUT_L1000_TE10_Start5_Val.mat"
            )  #
            data_list.append(MRFData)

        if "4SplineNoise11Flex" in type and "num250" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num250_4SplineNoise11Flex_D_LUT_L1000_TE10_Start5_Val.mat"
            )  #
            data_list.append(MRFData)

        # "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num100_1Spline5_D_LUT_L1000_TE10_Start5_Val.mat"
        self.is_split_range_T1T2 = is_split_range_T1T2 if "none" not in mode else False

        self.labels = data_list[0]["LUT"]
        self.D = data_list[0]["D"]
        self.RFpulses = data_list[0]["RFpulses"]

        for i in range(1, len(data_list)):
            self.labels = np.concatenate((self.labels, data_list[i]["LUT"]), 0)
            self.D = np.concatenate((self.D, data_list[i]["D"]), 0)
            self.RFpulses = np.concatenate((self.RFpulses, data_list[i]["RFpulses"]), 0)

        self.labels = torch.from_numpy(self.labels)
        self.D = torch.from_numpy(self.D[:, 0:L:subsamp])
        self.RFpulses = torch.from_numpy(self.RFpulses)

        # Already normalized in New... files
        # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        #  RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        # RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.

        self.TR = TR[0:L:subsamp].repeat(len(self.labels), 1).unsqueeze(1)
        self.TE = torch.ones(len(self.labels), 1, len_seq) * 10
        # self.RFpulses = RFpulses.repeat(len(self.labels), 1).unsqueeze(1)
        ###########################
        # # Add noise on RF

        # std1 = 0.3
        # std2 = 0.1
        # noise1 = torch.normal(0, std1, self.RFpulses[0, 0, :].shape).abs()
        # noise2 = torch.normal(0, std2, self.RFpulses[0, 0, :].shape).abs()
        # RFpulses = self.RFpulses + noise1
        # RFpulses2 = self.RFpulses + noise2

        # # Concat
        # self.RFpulses = torch.cat((RFpulses[: len(MRFData["LUT"])], RFpulses2[len(MRFData["LUT"]) :]), 0)

        # name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"
        # name2 = "1Spline5Flex"  #'2Spline11'#'2Spline11Flex' '5PieceConstant5Flex' '4SplineNoise11Flex' '1Spline5Flex'
        # file_name = name + name2 + ".mat"
        # traindata = sio.loadmat(file_name)
        # RFpulses = torch.tensor(traindata["rf"][100, :1000])
        # RFpulses = RFpulses * torch.pi / 180
        # RFpulses = RFpulses.repeat(len(self.labels), 1).unsqueeze(1)

        # name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"
        # name2 = "5PieceConstant5Flex"  #'2Spline11'#'2Spline11Flex' '5PieceConstant5Flex' '4SplineNoise11Flex' '1Spline5Flex'
        # file_name = name + name2 + ".mat"
        # traindata = sio.loadmat(file_name)
        # RFpulses2 = torch.tensor(traindata["rf"][100, :1000])
        # RFpulses2 = RFpulses2 * torch.pi / 180
        # RFpulses2 = RFpulses2.repeat(len(self.labels), 1).unsqueeze(1)

        # # Concat
        # self.RFpulses = torch.cat((RFpulses[: len(MRFData["LUT"])], RFpulses2[len(MRFData["LUT"]) :]), 0)
        ###########################

        if self.is_split_range_T1T2:
            # Split train and test set : T1 0~2500, 2500~5000
            T1 = self.labels[:, 0]
            T2 = self.labels[:, 1]
            condition = (
                (T1 < T1_condition_threshold) & (T2 < T2_condition_threshold)
                if "train" in mode
                else (T1 > T1_condition_threshold) & (T2 > T2_condition_threshold)
            )
            self.labels = self.labels[condition]
            self.D = self.D[condition]
            self.RFpulses = self.RFpulses[condition]
            self.TE = self.TE[condition]
            self.TR = self.TR[condition]

        # T1T2 scale
        self.T1 = self.labels[:, 0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)
        self.T2 = self.labels[:, 1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)

        if need_T1T2_logscale == True:

            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(self.T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(self.T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            self.D = self.D[T1T2_idx_n100]
            self.T1 = T1_log[T1T2_idx_n100]
            self.T2 = T2_log[T1T2_idx_n100]
            self.RFpulses = self.RFpulses[T1T2_idx_n100]
            self.TE = self.TE[T1T2_idx_n100]
            self.TR = self.TR[T1T2_idx_n100]

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR / 1000
            self.TE = self.TE / 1000

        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
            1, 2
        )  # (80100, 5, 200)

        # This if for Debug
        # self.labels = self.labels[0:20]
        # self.D = self.D[0:20]

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingImgDataset_Phantom(Dataset):
    """
    This get item by slice
    """

    # Input : T1T2, RF, TE, TR
    def __init__(self, mode="train"):
        """
        This data already done transformation
        So now : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        """

        # load file
        path = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
        if "test" in mode:
            # !!!!!!!!!!!!!!!!!!!!!!!!!
            sl_file = "New_test_10subject_4_5_slices_MRIs_new"  # "test_10subject_4_5_slices_MRIs"
            # !!!!!!!!!!!!!!!!!!!!!!!!!
        elif "train" in mode:
            sl_file = "New_train_10subject_4_5_slices_MRIs_new"  # "train_10subject_4_5_slices_MRIs"
        elif "none" in mode:
            # !!!!!!!!!!!!!!!!!!!!!!!!!
            sl_file = "New_all_10subject_4_5_slices_MRIs_new"  # "all_10subject_4_5_slices_MRIs"  # "all_MRIs" !!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!
        sl_file_name = path + sl_file
        # data_h5py = h5py.File(sl_file_name + ".h5", "r")
        data_h5py = np.load(sl_file_name + ".npz")
        self.D = data_h5py["X_all"]
        self.labels = data_h5py["labels"]

        # st = 130072
        # self.D = self.D[st : st + 1000]  # [0 : len(self.D) : 100]
        # self.labels = self.labels[st : st + 1000]  # [0 : len(self.labels) : 100]

        print("Data shape: ", self.D.shape, self.labels.shape)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingImgDataset_Phantom_RF(Dataset):
    """
    This get item by slice
    """

    # Input : T1T2, RF, TE, TR
    def __init__(self, mode="train"):
        """
        This data already done transformation
        So now : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        """

        # load file
        path = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
        sl_file = "num200_1Spline5Flex_New_all_10subject_4_5_slices_MRIs_new"
        sl_file_name = path + sl_file
        # data_h5py = h5py.File(sl_file_name + ".h5", "r")
        data_h5py = np.load(sl_file_name + ".npz")
        self.D = data_h5py["X_all"]
        self.labels = data_h5py["labels"]

        # st = 130072
        # self.D = self.D[st : st + 1000]  # [0 : len(self.D) : 100]
        # self.labels = self.labels[st : st + 1000]  # [0 : len(self.labels) : 100]

        print("Data shape: ", self.D.shape, self.labels.shape)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingImgDataset_real(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(
        self,
        need_T1T2_logscale=False,
        need_TETR_second=False,
        need_RF_degree=False,
    ):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """

        # Variable
        L = 1000
        subsamp = 1
        len_seq = L // subsamp

        # Load data
        MRFData = scipy.io.loadmat(
            "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/New_MRI_N128_L1000_TE10.mat"
        )

        # Data sequence
        # self.D = torch.from_numpy(np.real(MRFData["X_fullysamp"][:, 0:L:subsamp]))  # self.D:(128,128,1000)
        self.D = torch.from_numpy(MRFData["X_fullysamp"][:, 0:L:subsamp])  # self.D:(128,128,1000)
        # self.D = torch.flatten(self.D, 0, 1)  # self.D:(128*128,1000)
        # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        # Label
        len_seq = L // subsamp
        self.T1 = torch.flatten(torch.from_numpy(MRFData["T1_128"]), 0, 1)  # self.T1:(128*128,)
        self.T2 = torch.flatten(torch.from_numpy(MRFData["T2_128"]), 0, 1)  # self.T2:(128*128,)
        self.T1 = self.T1.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T1:(128*128,lenth, 1)
        self.T2 = self.T2.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T2:(128*128,lenth, 1)
        if need_T1T2_logscale == True:
            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(self.T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(self.T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            self.D = self.D[T1T2_idx_n100]
            self.T1 = T1_log[T1T2_idx_n100]
            self.T2 = T2_log[T1T2_idx_n100]

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        self.RFpulses = RFpulses.repeat(len(self.T1), 1).unsqueeze(1)
        self.TR = TR[0:L:subsamp].repeat(len(self.T1), 1).unsqueeze(1)
        self.TE = torch.ones(len(self.T1), 1, len_seq) * 10

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR / 1000
            self.TE = self.TE / 1000

        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
            1, 2
        )  # (128*128, 5, 1000)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


class PingImgDataset_real_noise(Dataset):
    # Input : T1T2, RF, TE, TR
    def __init__(self, need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False, type="num_250"):
        """
        Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
        Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
        """

        # Variable
        L = 1000
        subsamp = 1
        len_seq = L // subsamp

        # Load data
        data_list = []
        if "1Spline5Flex" in type and "num250" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num250_1Spline5Flex_MRI_N128_L1000_TE10.mat"
            )  #
            data_list.append(MRFData)
        if "1Spline5Flex" in type and "num200" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_1Spline5Flex_MRI_N128_L1000_TE10.mat"
            )  #
            data_list.append(MRFData)

        if "4SplineNoise11Flex" in type and "num200" in type:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_4SplineNoise11Flex_MRI_N128_L1000_TE10.mat"
            )  #
            data_list.append(MRFData)

        if type == "seqAll":
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_1Spline5Flex_MRI_N128_L1000_TE10.mat"
            )  #
            data_list.append(MRFData)

        if len(data_list) < 1:
            MRFData = scipy.io.loadmat(
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/num200_1Spline5Flex_MRI_N128_L1000_TE10.mat"
            )  #
            data_list.append(MRFData)

        self.T1 = data_list[0]["T1_128"]
        self.T2 = data_list[0]["T2_128"]
        self.D = data_list[0]["X_fullysamp"]
        self.RFpulses = data_list[0]["RFpulses"]

        for i in range(1, len(data_list)):
            self.T1 = np.concatenate((self.T1, data_list[i]["T1_128"]), 0)
            self.T2 = np.concatenate((self.T2, data_list[i]["T2_128"]), 0)
            self.D = np.concatenate((self.D, data_list[i]["X_fullysamp"]), 0)
            self.RFpulses = np.concatenate((self.RFpulses, data_list[i]["RFpulses"]), 0)

        self.T1 = torch.from_numpy(self.T1)
        self.T2 = torch.from_numpy(self.T2)
        self.D = torch.from_numpy(self.D[:, 0:L:subsamp])
        self.RFpulses = torch.from_numpy(self.RFpulses)

        self.T1 = self.T1.repeat(1, len_seq).unsqueeze(1)  # self.T1:(128*128,lenth, 1)
        self.T2 = self.T2.repeat(1, len_seq).unsqueeze(1)  # self.T2:(128*128,lenth, 1)

        # name = (
        #     "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/"
        #     + str(noise)
        #     + "noise_MRI_N128_L1000_TE10.mat"
        # )
        # MRFData = scipy.io.loadmat(name)

        # # Data sequence
        # # self.D = torch.from_numpy(np.real(MRFData["X_fullysamp"][:, 0:L:subsamp]))  # self.D:(128,128,1000)
        # self.D = torch.from_numpy(MRFData["X_fullysamp"][:, 0:L:subsamp])  # self.D:(128,128,1000)
        # # self.D = torch.flatten(self.D, 0, 1)  # self.D:(128*128,1000)
        # # self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

        # # Label
        # len_seq = L // subsamp
        # self.T1 = torch.flatten(torch.from_numpy(MRFData["T1_128"]), 0, 1)  # self.T1:(128*128,)
        # self.T2 = torch.flatten(torch.from_numpy(MRFData["T2_128"]), 0, 1)  # self.T2:(128*128,)
        # self.T1 = self.T1.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T1:(128*128,lenth, 1)
        # self.T2 = self.T2.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T2:(128*128,lenth, 1)
        if need_T1T2_logscale == True:
            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(self.T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(self.T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            self.D = self.D[T1T2_idx_n100]
            self.T1 = T1_log[T1T2_idx_n100]
            self.T2 = T2_log[T1T2_idx_n100]
            self.RFpulses = self.RFpulses[T1T2_idx_n100]

        # RF, TR, TE
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        # RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # self.RFpulses = torch.from_numpy(MRFData["RFpulses"]).unsqueeze(1)
        self.TR = TR[0:L:subsamp].repeat(len(self.T1), 1).unsqueeze(1)
        self.TE = torch.ones(len(self.T1), 1, len_seq) * 10

        if need_RF_degree == True:
            self.RFpulses = self.RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            self.TR = self.TR / 1000
            self.TE = self.TE / 1000

        # Concat
        self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
            1, 2
        )  # (128*128, 5, 1000)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.labels[idx]


# class PingImgDataset_Phantom_byslice(Dataset):
#     """
#     This get item by slice
#     """

#     # Input : T1T2, RF, TE, TR
#     def __init__(
#         self,
#     ):
#         """
#         This data already done transformation
#         So now : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
#         """
#         # n_slice_train = 80; n_slice_test = 22
#         # self.start_slice = 0 if 'train' in mode else n_slice_train; self.end_slice = n_slice_train-1 if 'train' in mode else n_slice_train-1+n_slice_test
#         self.len = 102 * 128 * 128
#         self.wh = 128 * 128

#     def __len__(self):
#         return self.len  # len(self.D)

#     def __getitem__(self, idx):
#         # index
#         slice_index = idx // (self.wh)  # from (12 or 10 slice*10subject=101total slice)
#         pixel_index = idx % (self.wh)  # from (128*128)

#         # load file
#         sl_file = "slice" + str(slice_index) + "_MRIs"
#         sl_file_name = (
#             "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_slice/" + sl_file
#         )
#         # d = np.load(sl_file_name + ".npz")
#         d = h5py.File(sl_file_name + ".h5", "r")

#         # get D and labels (already done T1T2, TETR, RF transformation)
#         D_slice = np.array(d.get("X_all"))
#         labels_slice = np.array(d.get("labels"))
#         return D_slice[pixel_index], labels_slice[pixel_index]


# class PingImgDataset_Phantom_byall(Dataset):
#     # Input : T1T2, RF, TE, TR
#     def __init__(self, mode="train"):
#         """
#         This data already done transformation
#         So now : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
#         """
#         # n_slice_train = 80; n_slice_test = 22
#         # self.start_slice = 0 if 'train' in mode else n_slice_train; self.end_slice = n_slice_train-1 if 'train' in mode else n_slice_train-1+n_slice_test

#         self.len = 81 * 128 * 128 if "tr" in mode else 21 * 128 * 128  # 102*128*128
#         self.split_th = 41 * 128 * 128
#         self.mode = mode

#         # load file
#         path = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_all/"
#         if "test" in mode:
#             sl_file = "test_all_MRIs"
#             sl_file_name = path + sl_file
#             self.d = np.load(sl_file_name + ".npz")
#         else:
#             sl_file_1 = "train_all_MRIs_1"
#             sl_file_2 = "train_all_MRIs_2"
#             sl_file_name_1 = path + sl_file_1
#             sl_file_name_2 = path + sl_file_2
#             self.d_1 = np.load(sl_file_name_1 + ".npz")
#             self.d_2 = np.load(sl_file_name_2 + ".npz")

#     def __len__(self):
#         return self.len  # len(self.D)

#     def __getitem__(self, idx):
#         if "train" in self.mode:
#             D = self.d_1 if idx > self.split_th else self.d_2
#             idx_ = idx if idx > self.split_th else idx - self.split_th
#         else:
#             D = self.d
#             idx_ = idx
#         return D["X_all_slice"][idx_], D["labels_slice"][idx_]

# def __getitem__(self, idx):
#     # index
#     slice_index = idx // (128 * 128)  # from (12 or 10 slice*10subject=101total slice)
#     pixel_index = idx % (128 * 128)  # from (128*128)

#     # load file
#     sl_file = "slice" + str(slice_index) + "_MRIs"
#     sl_file_name = (
#         "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_slice/" + sl_file
#     )
#     d = np.load(sl_file_name + ".npz")

#     # get D and labels (already done T1T2, TETR, RF transformation)
#     D_slice = d["X_all_slice"]
#     labels_slice = d["labels_slice"]
#     return D_slice[pixel_index], labels_slice[pixel_index]


# class PingImgDataset_Phantom(Dataset):
#     # Input : T1T2, RF, TE, TR
#     def __init__(
#         self,
#         mode="train",
#         is_split_range_T1T2=True,
#         need_T1T2_logscale=False,
#         need_TETR_second=False,
#         need_RF_degree=False,
#     ):
#         """
#         Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
#         Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
#         This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
#         """

#         subject_list = ["04", "05"] if "train" in mode else ["06", "18"]
#         num_slice = 1  # 12

#         L = 1000
#         subsamp = 1
#         len_seq = L // subsamp
#         T1_condition_threshold = 1000
#         T2_condition_threshold = 50

#         # Self
#         self.label = torch.zeros((len(subject_list)*))
#         self.D = {}

#         for n in range(len(subject_list)):
#             file = "subject" + str(subject_list[n]) + "_MRIs.mat"
#             file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/" + file
#             data_dict = mat73.loadmat(file_name)

#             for s in range(num_slice):

#                 # X_all(fully sampled image sequence)
#                 D_slice = torch.from_numpy(data_dict["X_all"][s][:, 0:L:subsamp])  # (128, 128, 1000)
#                 D_slice = torch.flatten(D_slice, 0, 1)  # (128*128, 1000)
#                 D_slice = torch.nn.functional.normalize(D_slice, p=2.0, dim=1)

#                 # LUT (T1,T2,PD)
#                 T1 = torch.from_numpy(data_dict["LUT"][0][s]).unsqueeze(-1)  # (128, 128, 1)
#                 T2 = torch.from_numpy(data_dict["LUT"][1][s]).unsqueeze(-1)  # (128, 128, 1)
#                 labels_slice = torch.cat((T1, T2), -1)  # (128, 128, 2)
#                 labels_slice = torch.flatten(labels_slice, 0, 1)  # (128*128, 2)

#                 # T1T2 scale
#                 T1 = labels_slice[:, 0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # (128*128, 1, 1000)
#                 T2 = labels_slice[:, 1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # (128*128, 1, 1000)
#                 if need_T1T2_logscale == True:
#                     T1 = torch.log10(T1)
#                     T2 = torch.log10(T2)

#                 # RFpulses : (128*128, 1, 1000)
#                 num_seq = len(labels_slice)
#                 RFpulses = torch.from_numpy(data_dict["Params"]["RFpulses"][0:L:subsamp])  # (1000,)
#                 RFpulses = RFpulses.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

#                 # TR : (128*128, 1, 1000)
#                 TR = torch.from_numpy(data_dict["Params"]["TR"][0:L:subsamp])  # (1000, )
#                 TR = TR.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

#                 # TE : (128*128, 1, 1000)
#                 TE = torch.ones(num_seq, 1, L // subsamp) * data_dict["Params"]["RFpulses"]  # (128*128, 1, 1000)

#                 # RFpulses, TR, TE scale
#                 if need_RF_degree == True:
#                     RFpulses = RFpulses * 180 / torch.pi

#                 if need_TETR_second == True:
#                     TR = TR / 1000
#                     TE = TE / 1000

#                 # Concat
#                 labels_slice = torch.cat((RFpulses, T1, T2, TE, TR), 1)  # (condition, 5, 1000)

#                 # Self
#                 # self.label[n][s] = labels_slice
#                 # self.D[n][s] = D_slice

#         # Self
#         # self.label[n, s] = labels_slice
#         # self.D[n, s] = D_slice

#     def __len__(self):
#         return len(self.D)

#     def __getitem__(self, idx):
#         return self.D[idx], self.labels[idx]


# class PingImgDataset(Dataset):
#     # Input : T1T2, RF, TE, TR
#     def __init__(self, mode="test", need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False):
#         """
#         Approx RNN input : (need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True)
#         Our Bloch decoder input : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
#         This data : (need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False)
#         """
#         MRFData = scipy.io.loadmat("lightning_bolts/Bloch_decoder/data/Pingfan/MRI_N128_L1000_TE10.mat")
#         L = 1000
#         subsamp = 1
#         self.D = torch.from_numpy(np.real(MRFData["X_fullysamp"][:, 0:L:subsamp]))  # self.D:(128,128,1000)
#         self.D = torch.flatten(self.D, 0, 1)  # self.D:(128*128,1000)
#         self.D = torch.nn.functional.normalize(self.D, p=2.0, dim=1)

#         # Load data - shape : (# of subject, # of slices, 128, 128, L=1000)
#         self.D = []
#         subject_list = ["04"]  # ['04', '05', '06', '18']
#         for i in subject_list:
#             file = "subject" + str(i) + "_MRIs.mat"
#             file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/" + file
#             data_dict = mat73.loadmat(file_name)

#         # T1T2 scale
#         len_seq = L // subsamp
#         self.T1 = torch.flatten(torch.from_numpy(MRFData["T1_128"]), 0, 1)  # self.T1:(128*128,)
#         self.T2 = torch.flatten(torch.from_numpy(MRFData["T2_128"]), 0, 1)  # self.T2:(128*128,)
#         self.T1 = self.T1.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T1:(128*128,lenth, 1)
#         self.T2 = self.T2.unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # self.T2:(128*128,lenth, 1)
#         if need_T1T2_logscale == True:
#             self.T1 = torch.log10(self.T1)
#             self.T2 = torch.log10(self.T2)

#         # RF, TR, TE
#         RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
#         RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
#         # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
#         self.RFpulses = RFpulses.repeat(len(self.T1), 1).unsqueeze(1)
#         self.TR = TR[0:L:subsamp].repeat(len(self.T1), 1).unsqueeze(1)
#         self.TE = torch.ones(len(self.T1), 1, len_seq) * 10

#         if need_RF_degree == True:
#             self.RFpulses = self.RFpulses * 180 / torch.pi

#         if need_TETR_second == True:
#             self.TR = self.TR / 1000
#             self.TE = self.TE / 1000

#         # Concat
#         self.labels = torch.cat((self.RFpulses, self.T1, self.T2, self.TE, self.TR), 1).transpose(
#             1, 2
#         )  # (128*128, 5, 1000)

#         # Simpler
#         # self.labels = self.labels[0:20]
#         # self.D = self.D[0:20]

#     def __len__(self):
#         return len(self.D)

#     def __getitem__(self, idx):
#         return self.D[idx], self.labels[idx]
