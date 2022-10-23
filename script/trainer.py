from errno import ESTALE
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import relu
import numpy as np
from torch.autograd import Variable

import neptune.new as neptune
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Bloch_decoder.top_MRI_simulator import Top_MRI_simulator
from Bloch_decoder.utils.generate_RF_TR import generate_RF_TR
from Bloch_decoder.All_simulator import Top_MRI_simulator as Top_MRI_simulator_2
from Bloch_decoder.Bloch_equation_jit import Top_MRI_simulator as Top_MRI_simulator_3


import torch.nn as nn


def conv1d(in_planes, out_planes, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding="same", bias=True)


class GRU(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_size_1=32,
        hidden_size_2=32,
        hidden_size_3=32,
        output_size=3,
        num_layers=1,
        device="cuda",
    ):
        super(GRU, self).__init__()
        """
        # This model is trained in the paper: 
        # output : (m, dm/dT1, dm/dT2)
        # input : (RF, T1, T2, TE, TR)
        """
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.num_layers = num_layers
        self.device = device

        self.gru_1 = nn.GRU(input_size, hidden_size_1, num_layers, batch_first=True)
        self.gru_2 = nn.GRU(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
        self.gru_3 = nn.GRU(hidden_size_1, hidden_size_3, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size_3, output_size)

    def forward(self, x, h_set):
        input_X = x
        out_gru_1, h_set[0] = self.gru_1(input_X, h_set[0])  # h_1
        out_gru_2, h_set[1] = self.gru_2(out_gru_1, h_set[1])  # h_2
        out_gru_3, h_set[2] = self.gru_2(out_gru_2, h_set[2])  # h_3
        out_Dense_out = self.fc_out(out_gru_3)

        return out_Dense_out, h_set

    def init_hidden_set(self):
        h_set = {}
        h_set[0] = torch.zeros(self.num_layers, self.hidden_size_1, device=self.device)  # h_1
        h_set[1] = torch.zeros(self.num_layers, self.hidden_size_2, device=self.device)  # h_2
        h_set[2] = torch.zeros(self.num_layers, self.hidden_size_3, device=self.device)  # h_3
        return h_set


class our_GRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=1, num_layers=3, device="cuda"):
        super(our_GRU, self).__init__()
        """
        # This model is our new model
        # output : (N, L, Out)
        # input : (RF, T1, T2, TE, TR) : batch, length, in_dim(input_size=5)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # (N, L, H)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):  # This is 'many-to-many'
        input_X = x
        out_gru_1, h = self.gru(input_X, h)  # out: (Batch, lengh, in), h: (layer, batch, hidden_size)
        out_Dense_out = self.fc_out(out_gru_1)
        return out_Dense_out

    def forward_many_to_one(self, x, h):
        ##################
        # Example for MNIST:
        #   sequence_length = 28
        #   input_size=28
        #   hidden_size=128
        #   num_layers=2
        #   batch_size=100
        #   output_size=10
        ##################
        input_X = x
        out_gru_1, h = self.gru(input_X, h)  # out: (Batch, lengh, in), h: (layer, batch, hidden_size)
        # Decode the hidden state of the last time step
        out_Dense_out = self.fc_out(
            out_gru_1[:, -1, :]
        )  # out_Dense_out: tensor of shape (batch_size, seq_length, hidden_size)
        return out_Dense_out

    def init_hidden_set(self, batch_size=200):
        h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)  # h_1
        return h_init


class FC_Encoder(nn.Module):  # fc1 :200 200 fc2:200 300 fc3: 300 2
    def __init__(self, input_prod_size=1000, enc_out_dim=300, latent_dim=2):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_prod_size, input_prod_size)  # .to(torch.complex64)
        self.fc2 = nn.Linear(input_prod_size, enc_out_dim)  # .to(torch.complex64)
        self.fc3 = nn.Linear(enc_out_dim, enc_out_dim)  # .to(torch.complex64)
        self.linear = nn.Linear(enc_out_dim, latent_dim)  # .to(torch.cfloat)  # complex64

    def forward(self, x):
        h1 = F.relu(
            self.fc1(x)
        )  # F.softplus(self.fc1(x), beta=1)  # real_imaginary_relu(self.fc1(x))  # F.softplus(self.fc1(x), beta=1)  #
        h2 = F.relu(self.fc2(h1))  # F.softplus(self.fc1(h1), beta=1)
        h3 = F.relu(self.fc3(h2))  # F.softplus(self.fc1(h1), beta=1)
        return self.linear(h3)  # F.relu(self.linear(h2))


class RNN_Encoder(nn.Module):
    # This RNN Encoder is 'many to one': from sequence predict T1T2
    def __init__(
        self, input_prod_size=1000, enc_out_dim=300, latent_dim=2
    ):  # L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        super(RNN_Encoder, self).__init__()
        self.simulator = our_GRU(
            input_size=input_prod_size,
            hidden_size=enc_out_dim,  # 32
            output_size=latent_dim,
            num_layers=3,
        )
        self.simulator.cuda()

    def forward(self, x):
        """
        # x : (n, length) -> (n, 1, length)
        # out : (n, 2) T1T2
        """
        x = x.unsqueeze(1)
        h_init = self.simulator.init_hidden_set(batch_size=len(x))
        pred_T1T2 = self.simulator.forward_many_to_one(x, h_init)
        return pred_T1T2


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode="embedded", dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ["gaussian", "embedded", "dot", "concatenate"]:
            raise ValueError("`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`")

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels),
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1), nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class Hydra_Encoder(nn.Module):
    def __init__(self):
        super(Hydra_Encoder, self).__init__()
        self.c1 = conv1d(1, 16, kernel_size=21)
        self.c2 = conv1d(16, 16, kernel_size=21)
        self.mp1 = nn.MaxPool1d(2)
        self.non_local_block_32 = NLBlockND(in_channels=32, mode="embedded", dimension=1, bn_layer=False)
        self.non_local_block_64 = NLBlockND(in_channels=64, mode="embedded", dimension=1, bn_layer=False)
        self.non_local_block_128 = NLBlockND(in_channels=128, mode="embedded", dimension=1, bn_layer=False)
        self.linear = nn.Linear(500, 2)

        self.res1 = conv1d(in_planes=16, out_planes=32, kernel_size=1)
        self.res2 = conv1d(in_planes=32, out_planes=32, kernel_size=21)
        self.res3 = conv1d(in_planes=32, out_planes=64, kernel_size=1)
        self.res4 = conv1d(in_planes=64, out_planes=64, kernel_size=21)
        self.res5 = conv1d(in_planes=64, out_planes=128, kernel_size=1)
        self.res6 = conv1d(in_planes=128, out_planes=128, kernel_size=21)

    def forward(self, x):
        return self.resnet_MRF(x)

    def resnet_layer(
        self,
        x,
        model,
        activation="relu",
        batch_normalization=False,
        conv_first=True,
    ):
        """1D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input 1D signal
            num_filters (int): Conv1D number of filters
            kernel_size (int): Conv1D kernel dimensions
            strides (int): Conv1D stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        if conv_first:
            x = model(x)

            if activation is not None:
                x = relu(x)
        else:  # full pre-activation

            if activation is not None:
                x = relu(x)
            x = model(x)
        return x

    def resnet_MRF(self, x):
        # create model CNN
        """ResNet Version 1 Model builder
        Stacks of BN-ReLU-Conv1D
        # Arguments
            input_shape (tensor): shape of input image tensor, e.g. 200x1
            depth (int): number of core convolutional layers, 6n+2, e.g. 20, 32, 44
            num_classes (int): number of classes, e.g.2 types of tissue parameters,
            T1 and T2)
        # Returns
            model (Model): Keras model instance
        """

        # Shape of input

        # Forward

        # shape change
        x = x.unsqueeze(1)

        # conv1d relu
        # conv1d relu
        # non local block
        # maxplooling

        x = self.c1(x)
        x = self.c2(x)
        x = self.mp1(x)

        # x = resnet_layer
        # y = resnet_layer
        # x+y
        # non local block
        # maxplling

        x = self.resnet_layer(x, self.res1)
        y = self.resnet_layer(x, self.res2)
        x = x + y
        x = self.non_local_block_32(x)  # mode: `embedded`, `gaussian`, `dot` or `concatenate`.

        x = self.resnet_layer(x, self.res3)
        y = self.resnet_layer(x, self.res4)
        x = x + y
        x = self.non_local_block_64(x)  # mode: `embedded`, `gaussian`, `dot` or `concatenate`.

        x = self.resnet_layer(x, self.res5)
        y = self.resnet_layer(x, self.res6)
        x = x + y
        x = self.non_local_block_128(x)  # mode: `embedded`, `gaussian`, `dot` or `concatenate`.

        # Gloval average pooling
        # Fully connected
        x = torch.mean(x, 1)
        x = self.linear(x)
        return x


class FC_Decoder(nn.Module):
    def __init__(self, input_prod_size=1000, enc_out_dim=300, latent_dim=2):
        super(FC_Decoder, self).__init__()
        self.linear_ = nn.Linear(latent_dim, enc_out_dim)  # .to(torch.complex64)
        self.fc4 = nn.Linear(enc_out_dim, enc_out_dim)  # .to(torch.complex64)
        self.fc5 = nn.Linear(enc_out_dim, input_prod_size)  # .to(torch.complex64)
        self.fc6 = nn.Linear(input_prod_size, input_prod_size)  # .to(torch.complex64)

    def forward(self, z):
        h4 = F.relu(
            self.linear_(z)
        )  # F.softplus(self.fc1(x), beta=1)  # real_imaginary_relu(self.fc1(x))  # F.softplus(self.fc1(x), beta=1)  #
        h5 = F.relu(self.fc4(h4))  # F.softplus(self.fc1(h1), beta=1)
        h6 = F.relu(self.fc5(h5))  # F.softplus(self.fc1(h1), beta=1)
        return self.fc6(h6)  # torch.sigmoid(self.fc4(h3))


class RNN_Decoder:
    def __init__(self):  # L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        self.simulator = our_GRU()
        self.simulator.cuda()
        # self.simulator.eval()

    def RNN_forward(self, z):
        z = z.transpose(1, 2)  # z : (batch, length, 5)  transpose(0,1).transpose(0,2)
        h_init = self.simulator.init_hidden_set(batch_size=len(z))
        seq = self.simulator.forward(z, h_init)
        seq = seq.squeeze(-1)
        return seq

    def forward(self, batch):
        """
        # z : (n,5, length)- 5: RFpulses, T1, T2, TE, TR
        # x : (n, length)
        # out : (n, length)
        """
        x, z = batch
        # z = z.transpose(1,2) # z : (batch, length, 5)  transpose(0,1).transpose(0,2)
        h_init = self.simulator.init_hidden_set(batch_size=len(z))
        seq = self.simulator.forward(z, h_init)
        seq = seq.squeeze(-1)
        return seq


class BlochImgEqDecoder:
    def __init__(self, L=1000, subsamp=1, TE=10):
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        TR = TR[0:L:subsamp]

        self.TR = TR
        self.RFpulses = RFpulses
        self.TE = TE

    def forward(self, decoder_input):
        T1_image = decoder_input[0, :, :]
        T2_image = decoder_input[1, :, :]
        PD_image = decoder_input[2, :, :]
        simulator = Top_MRI_simulator()
        MRIs = simulator.build_fully_sampled_contrasts_modified(
            T1_image=T1_image,
            T2_image=T2_image,
            TE=self.TE,
            RFpulses=self.RFpulses,
            TR=self.TR,
            PD_image=PD_image,
        )
        print(MRIs.shape)
        # print(MRIs.transpose(1, 2).transpose(0, 1).shape)
        print("6666" * 30)
        return MRIs  # .transpose(1, 2).transpose(0, 1)


class BlochSeqEqDecoder:
    def __init__(self, L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        self.L = L // subsamp
        self.TE = TE

        # Real value
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        TR = TR[0:L:subsamp]

        self.TR = TR
        self.RFpulses = RFpulses

    def forward(self, decoder_input):
        # Decoder_input : (n,2)
        T1_values = decoder_input[:, 0]
        T2_values = decoder_input[:, 1]
        simulator = Top_MRI_simulator()
        D, LUT = simulator.build_dictionary_fisp_seq(
            T1_values=T1_values, T2_values=T2_values, L=self.L, TE=self.TE, RFpulses=self.RFpulses, TR=self.TR
        )
        return D


class BlochSeqEqDecoder_temp:
    def __init__(self, L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        self.L = L // subsamp
        self.TE = TE

        # Real value
        RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        TR = TR[0:L:subsamp]

        self.TR = TR
        self.RFpulses = RFpulses

    def forward(self, batch):
        # Decoder_input : (n,2)
        x, z = batch
        T1_values = z[:, 1, 0]
        T2_values = z[:, 2, 0]
        simulator = Top_MRI_simulator()
        D, LUT = simulator.build_dictionary_fisp_seq(
            T1_values=T1_values, T2_values=T2_values, L=self.L, TE=self.TE, RFpulses=self.RFpulses, TR=self.TR
        )
        return D


class BlochSeqEqDecoder_ver2:
    def __init__(self):  # larger subsamp makes shorter length
        self.L = 1000

    def forward(self, batch):

        # variables
        x, z = batch
        T1_values = z[:, 0, 1]  # T1: (n,)
        T2_values = z[:, 0, 2]  # T2: (n,)
        TE = z[:, :, 3]  # TE: (n, length)
        TR = z[:, :, 4]  # TR: (n, length)
        RF = z[:, :, 0]  # RF: (n, length)
        simulator = Top_MRI_simulator_3()
        D, LUT = simulator.build_dictionary_fisp_seq(
            T1_values=T1_values, T2_values=T2_values, L=self.L, TE=TE, RFpulses=RF, TR=TR
        )
        # D = torch.real(D)

        # ########
        # print(T1_values[2:5])
        # target = torch.ones_like(D).cuda()
        # loss = (D - target).pow(2).mean()
        # loss.backward(retain_graph=True)
        # print(self.encoder.simulator.fc_out.weight.grad.sum())
        # print()
        # ##########
        return D


class BlochSeqEqDecoder_RNN_from_paper:
    def __init__(self):  # L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        # self.L = L // subsamp
        # self.TE = TE

        # # Real value
        # RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
        # RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
        # RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
        # TR = TR[0:L:subsamp]

        # self.TR = TR
        # self.RFpulses = RFpulses

        self.simulator = GRU()
        self.simulator.load_state_dict(torch.load("./lightning_bolts/script/new_pytorch_RNN_EPG.pt"))
        self.simulator.cuda()
        self.simulator.eval()

    def forward(self, batch):
        """
        # z : (n,5)-RFpulses, T1, T2, TE, TR
        # x : (n, 1129) or (n, 200)
        # out : (n, 1129) or (n, 200)
        """
        x, z = batch
        seq = []
        h_set = self.simulator.init_hidden_set()
        for i in range(z.shape[2]):
            inp = z[:, :, i]
            out, h_set = self.simulator.forward(inp, h_set)
            seq.append(out[:, 0].unsqueeze(1))  # Only save m. (output of simulator : [m, dm_dT1, dm_dT2])
        seq = torch.cat(seq, 1)
        return seq


def model_load_matcher(file_name):
    """
    copy linux:
    cp output/SeqBlochDecoder/SEQ-310/last.ckpt lightning_bolts/script/RNN_model/RNN_epoch200_SEQ_310.ckpt
    """
    state_dict = torch.load(file_name)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    return state_dict


class BlochSeqEqDecoder_our_RNN:
    def __init__(self, model_dir):  # L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
        self.simulator = our_GRU()
        self.simulator.load_state_dict(model_load_matcher(model_dir))
        self.simulator.cuda()
        # self.simulator.eval()

    # def RNN_forward(self, z):
    #     z = z.transpose(1, 2)  # z : (batch, length, 5)  transpose(0,1).transpose(0,2)
    #     h_init = self.simulator.init_hidden_set(batch_size=len(z))
    #     seq = self.simulator.forward(z, h_init)
    #     seq = seq.squeeze(-1)
    #     return seq

    def forward(self, batch):
        """
        # z : (n,length, 5)- 5: RFpulses, T1, T2, TE, TR
        # x : (n, length)
        # out : (n, length)
        """
        x, z = batch
        # z = z.transpose(1, 2)  # z : (batch, length, 5)  transpose(0,1).transpose(0,2)
        h_init = self.simulator.init_hidden_set(batch_size=len(z))
        seq = self.simulator.forward(z, h_init)
        seq = seq.squeeze(-1)

        return seq


class LitTrainApproxModel(pl.LightningModule):
    def __init__(self, model_type="RNN", is_input_RF=1):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        input_size = 5 if is_input_RF > 0 else 2
        if "RNN" in model_type:
            self.model = our_GRU(input_size=input_size)
        else:
            self.model = FC_Decoder(latent_dim=input_size)

    def RNN_forward(self, z):
        # z : (batch, length, 5)  transpose(0,1).transpose(0,2)
        h_init = self.model.init_hidden_set(batch_size=len(z))
        seq = self.model.forward(z, h_init)
        seq = seq.squeeze(-1)
        return seq

    def step(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        if "RNN" in self.model_type:
            x_hat = self.RNN_forward(z)
        else:
            x_hat = self.model.forward(z)
        loss = F.mse_loss(x_hat, x, reduction="mean")
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        input_prod_size,
        enc_out_dim,
        latent_dim,
        rec_lambda=1,
        decoder_type="our_RNN_bloch",
        encoder_type="FC",
        is_emb_loss=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.rec_lambda = rec_lambda
        self.decoder_type = decoder_type
        self.is_emb_loss = is_emb_loss
        if "RNN" in encoder_type:
            self.encoder = RNN_Encoder(input_prod_size, enc_out_dim, latent_dim)
        elif "Hydra" in encoder_type:
            self.encoder = Hydra_Encoder()
        else:
            self.encoder = FC_Encoder(input_prod_size, enc_out_dim, latent_dim)

        # none : only encoder
        if "none" in decoder_type or "None" in decoder_type:
            self.type = "none"
            self.step = self.step_en
            self.decoder = None

        # simple : encoder+decoder (update both models)
        if "simple" in decoder_type:
            if "RNN" in decoder_type:
                self.type = "simple_RNN"
                self.step = self.step_en_de_RNN
                self.decoder = RNN_Decoder()
            else:
                self.type = "simple_FC"
                self.step = self.step_en_de
                self.decoder = FC_Decoder()

        # our_RNN : encoder + causal decoder (only update encoder)
        if "our_RNN" in decoder_type or "causal" in decoder_type:
            self.type = "causal_RNN"

            # # Approxi
            # exp_id = "SPLIT-13"
            # model_dir = "/mnt/ssd/jj/Research/cauMedical/output/Split/" + str(exp_id) + "/last.ckpt"
            # self.step = self.step_en_de_RNN
            # self.decoder = BlochSeqEqDecoder_our_RNN(model_dir)

            # Bloch equation
            self.step = self.step_en_de_Bloch
            self.decoder = BlochSeqEqDecoder_ver2()

    def on_after_backward(self):
        # simulator.gru.weight_ih_l0
        # simulator.gru.weight_hh_l0
        # simulator.gru.bias_ih_l0
        # simulator.gru.bias_hh_l0
        # simulator.gru.weight_ih_l1
        # simulator.gru.weight_hh_l1
        # simulator.gru.bias_ih_l1
        # simulator.gru.bias_hh_l1
        # simulator.gru.weight_ih_l2
        # simulator.gru.weight_hh_l2
        # simulator.gru.bias_ih_l2
        # simulator.gru.bias_hh_l2
        # simulator.fc_out.weight
        # simulator.fc_out.bias
        # for name, param in self.encoder.named_parameters():
        #     print(name)
        # print(self.encoder.simulator.fc_out.weight.grad.sum())
        pass

    def step_en_de_RNN(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 4].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # encoder
        T1T2_hat = self.encoder.forward(x)  # T1T2_hat: (n,2)
        T1T2_hat_ = T1T2_hat.unsqueeze(-1).repeat(1, 1, TE.shape[1]).transpose(1, 2)  # T1T2_hat: (n, length, 2)

        # decoder
        z_hat = torch.cat((RF, T1T2_hat_, TE, TR), 2)
        x_hat = self.decoder.forward((x, z_hat)).squeeze(-1)

        # loss
        # rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        rec_loss = (x_hat - x).pow(2).mean()
        emb_loss = torch.zeros(1)
        if self.is_emb_loss > 0:
            emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
            loss = emb_loss + (self.rec_lambda * rec_loss)
        else:
            loss = rec_loss

        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": rec_loss.type(torch.float),
                f"{mode}_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en_de(x, T1T2, x_hat, T1T2_hat, batch_idx)
        return loss

    def step_en_de_Bloch(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x_ori, z = batch
        x = x_ori.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 4].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # encoder
        T1T2_hat = self.encoder.forward(x)  # T1T2_hat: (n,2)
        T1T2_hat_ = T1T2_hat.unsqueeze(-1).repeat(1, 1, TE.shape[1]).transpose(1, 2)  # T1T2_hat: (n, length, 2)

        # decoder
        z_hat = torch.cat((RF, T1T2_hat_, TE, TR), 2)
        x_hat = self.decoder.forward((x, z_hat)).squeeze(-1)

        # # Get rid of outlier
        # min_v = x.min()
        # max_v = x.max()
        # no_ind_li = []
        # for i in range(len(x_hat)):
        #     con = torch.real(x_hat[i]) > min_v
        #     con_2 = torch.real(x_hat[i]) < max_v
        #     if torch.all(con) and torch.all(con_2):
        #         no_ind_li.append(i)
        # x_hat = x_hat[no_ind_li]
        # x = x[no_ind_li]

        # loss
        # rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        rec_loss = (x_hat - x_ori).pow(2).mean()
        emb_loss = torch.zeros(1)
        if self.is_emb_loss > 0:
            emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
            loss = emb_loss + (self.rec_lambda * rec_loss)
        else:
            loss = rec_loss

        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": rec_loss.type(torch.float),
                f"{mode}_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en_de(x, T1T2, x_hat, T1T2_hat, batch_idx)
        return loss

    def step_en_de(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 4].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        x_hat = self.decoder.forward(T1T2_hat)
        x_hat = x_hat.real
        # rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        rec_loss = (x_hat - x).pow(2).mean()
        emb_loss = torch.zeros(1)

        if self.is_emb_loss > 0:
            emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
            loss = emb_loss + (self.rec_lambda * rec_loss)
        else:
            loss = rec_loss
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": rec_loss.type(torch.float),
                f"{mode}_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en_de(x, T1T2, x_hat, T1T2_hat, batch_idx)
        return loss

    def step_en(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 4].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": 0.0,
                f"{mode}_emb_loss": loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en(x, T1T2, T1T2_hat, batch_idx)
        return loss

    def save_en_de(self, x, z, x_hat, z_hat, batch_idx):
        mode = "test"
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            x_hat=x_hat.detach().cpu(),
            x=x.detach().cpu(),
            z_hat=z_hat.detach().cpu(),
            z=z.cpu(),
        )

    def save_en(self, x, z, z_hat, batch_idx):
        mode = "test"
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            x=x.detach().cpu(),
            z_hat=z_hat.detach().cpu(),
            z=z.cpu(),
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        # if dataloader_idx == 0:
        #     mode = "val_seqTe"

        self.step(batch["seqTe"], batch_idx, mode="val_seqTe")
        # self.step(batch["phantomTe"], batch_idx, mode="val_phantomTe")
        self.step(batch["phantomAll"], batch_idx, mode="val_phantomAll")
        # self.step(batch["phantomAll_noise"], batch_idx, mode="val_phantomAll_noise")
        self.step(batch["real"], batch_idx, mode="val_real")
        # self.step(batch["seqTe_RF"], batch_idx, mode="val_seqTe_RF")
        # self.step(batch["real_noise"], batch_idx, mode="val_real_noise")

    def test_step(self, batch, batch_idx):
        self.step(batch["seqTe"], batch_idx, mode="test_seqTe")
        # self.step(batch["phantomTe"], batch_idx, mode="test_phantomTe")
        self.step(batch["phantomAll"], batch_idx, mode="test_phantomAll")
        # self.step(batch["phantomAll_noise"], batch_idx, mode="test_phantomAll_noise")
        self.step(batch["real"], batch_idx, mode="test_real")
        # self.step(batch["real_noise"], batch_idx, mode="test_real_noise")

    def configure_optimizers(self):
        both_update = ["none", "simple_RNN", "simple_FC"]
        encoder_update_only = ["causal_RNN", "our_RNN"]  # Only update encoder, Not update pre-traiend RNN

        return torch.optim.Adam(self.parameters(), lr=1e-3)
        # if self.type in both_update:
        #     return torch.optim.Adam(self.parameters(), lr=1e-3)
        # elif self.type in encoder_update_only:
        #     return torch.optim.Adam(self.encoder.parameters(), lr=1e-3)


class LitModel_backup(pl.LightningModule):
    def __init__(
        self, input_prod_size, enc_out_dim, latent_dim, rec_lambda=1, decoder_type="our_RNN_bloch", is_emb_loss=1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.rec_lambda = rec_lambda
        self.decoder_type = decoder_type
        self.is_emb_loss = is_emb_loss
        self.encoder = FC_Encoder(input_prod_size, enc_out_dim, latent_dim)

        # none : only encoder
        if "none" in decoder_type or "None" in decoder_type:
            self.type = "none"
            self.step = self.step_en
            self.decoder = None

        # simple : encoder+decoder (update both models)
        if "simple" in decoder_type:
            if "RNN" in decoder_type:
                self.type = "simple_RNN"
                self.step = self.step_en_de_RNN
                self.decoder = RNN_Decoder()
            else:
                self.type = "simple_FC"
                self.step = self.step_en_de
                self.decoder = FC_Decoder()

        # our_RNN : encoder + causal decoder (only update encoder)
        if "our_RNN" in decoder_type or "causal" in decoder_type:
            self.type = "causal_RNN"
            exp_id = "SPLIT-13"
            model_dir = "/mnt/ssd/jj/Research/cauMedical/output/Split/" + str(exp_id) + "/last.ckpt"
            self.step = self.step_en_de_RNN
            self.decoder = BlochSeqEqDecoder_our_RNN(model_dir)

    def step_en_de_RNN(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # encoder
        T1T2_hat = self.encoder.forward(x)  # T1T2_hat: (n,2)
        T1T2_hat_ = T1T2_hat.unsqueeze(-1).repeat(1, 1, TE.shape[1]).transpose(1, 2)  # T1T2_hat: (n, length, 2)

        # decoder
        z_hat = torch.cat((RF, T1T2_hat_, TE, TR), 2)
        x_hat = self.decoder.forward((x, z_hat)).squeeze(-1)

        # loss
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        emb_loss = torch.zeros(1)
        if self.is_emb_loss > 0:
            emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
            loss = emb_loss + (self.rec_lambda * rec_loss)
        else:
            loss = rec_loss

        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": rec_loss.type(torch.float),
                f"{mode}_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en_de(x, T1T2, x_hat, T1T2_hat, batch_idx)
        return loss

    def step_en_de(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        x_hat = self.decoder.forward(T1T2_hat)
        x_hat = x_hat.real
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        emb_loss = torch.zeros(1)

        if self.is_emb_loss > 0:
            emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
            loss = emb_loss + (self.rec_lambda * rec_loss)
        else:
            loss = rec_loss
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": rec_loss.type(torch.float),
                f"{mode}_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en_de(x, T1T2, x_hat, T1T2_hat, batch_idx)
        return loss

    def step_en(self, batch, batch_idx, mode="test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": 0.0,
                f"{mode}_emb_loss": loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if mode == "test" and batch_idx < 5:
            self.save_en(x, T1T2, T1T2_hat, batch_idx)
        return loss

    def save_en_de(self, x, z, x_hat, z_hat, batch_idx):
        mode = "test"
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            x_hat=x_hat.detach().cpu(),
            x=x.detach().cpu(),
            z_hat=z_hat.detach().cpu(),
            z=z.cpu(),
        )

    def save_en(self, x, z, z_hat, batch_idx):
        mode = "test"
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            x=x.detach().cpu(),
            z_hat=z_hat.detach().cpu(),
            z=z.cpu(),
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        both_update = ["none", "simple_RNN", "simple_FC"]
        encoder_update_only = ["causal_RNN"]  # Only update encoder, Not update pre-traiend RNN

        if self.type in both_update:
            return torch.optim.Adam(self.parameters(), lr=1e-3)
        elif self.type in encoder_update_only:
            return torch.optim.Adam(self.encoder.parameters(), lr=1e-3)


class LitModel_Test(pl.LightningModule):
    def __init__(
        self,
        exp_id,
        input_prod_size,
        enc_out_dim,
        latent_dim,
        test_data_type,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.step = self.step_en

        # self.encoder.load_state_dict(model_load_matcher(model_dir)); model_dir = "/mnt/ssd/jj/Research/cauMedical/output/Split/" + str(exp_id) + "/last.ckpt"

        self.ckpt = self.load_ckpt(exp_id)
        self.encoder, is_hydra = self.check_encoder_type(self.ckpt)  # (input_prod_size, enc_out_dim, latent_dim)
        if is_hydra < 1:
            self.encoder = self.encoder(input_prod_size, enc_out_dim, latent_dim).cuda()
        else:
            self.encoder = self.encoder().cuda()
        self.safe_model_loader(self.encoder, self.ckpt)
        self.encoder = self.encoder.cuda()
        self.encoder.eval()

    def check_encoder_type(self, ckpt):
        is_hydra = 0
        if "gru" in list(ckpt["state_dict"].keys())[0]:
            encoder = RNN_Encoder
        elif "encoder.c1.weight" in list(ckpt["state_dict"].keys())[0]:
            encoder = Hydra_Encoder
            is_hydra = 1
        else:
            encoder = FC_Encoder
        return encoder, is_hydra

    def safe_model_loader(self, model, ckpt):
        new_ckpt = {}
        for i in list(ckpt["state_dict"].keys()):
            if "encoder" in i:
                model_parameter_name = i[8:]
                new_ckpt[str(model_parameter_name)] = ckpt["state_dict"][i]
        model.load_state_dict(new_ckpt)
        return

    def load_ckpt(self, exp_id):
        # path = os.path.join(root_dir, exp_id, "last.ckpt")
        path = "/mnt/ssd/jj/Research/cauMedical/output/Split/" + str(exp_id) + "/last.ckpt"
        return torch.load(path, map_location="cuda")

    def step_en(self, batch, batch_idx, mode="img_test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": 0.0,
                f"{mode}_emb_loss": loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if "test" in mode and batch_idx < 5:
            self.save_en(T1T2, T1T2_hat, batch_idx, mode)
        return loss

    def step_en_for_jupyter(self, batch, batch_idx):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
        return T1T2_hat, T1T2, loss

    def save_en(self, z, z_hat, batch_idx, mode):
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            z_hat=z_hat.float().detach().cpu(),
            z=z.float().cpu(),
        )

    def validation_step(self, batch, batch_idx):
        mode_name = str(self.hparams.test_data_type) + "_val"
        self.step(batch, batch_idx, mode=mode_name)

    def test_step(self, batch, batch_idx):
        self.step(batch["seqTe"], batch_idx, mode="test_seqTe")
        self.step(batch["phantomTe"], batch_idx, mode="test_phantomTe")
        self.step(batch["phantomAll"], batch_idx, mode="test_phantomAll")
        self.step(batch["real"], batch_idx, mode="test_real")


class LitDictionary_baseline_Test(pl.LightningModule):
    def __init__(
        self,
        train_data_type,
        test_data_type,
        data_module,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Get and save D, LUT
        if "phantomTr" in train_data_type:
            file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/train_10subject_4_5_slices_MRIs.npz"
            self.dictionary = torch.from_numpy(np.load(file_name)["X_all"]).t()
            self.LUT_T1T2 = torch.from_numpy(np.load(file_name)["labels"])[:, 1:3, 0]
        elif "phantomAll" in train_data_type:
            file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/all_10subject_4_5_slices_MRIs.npz"
            self.dictionary = torch.from_numpy(np.load(file_name)["X_all"]).t()
            self.LUT_T1T2 = torch.from_numpy(np.load(file_name)["labels"])[:, 1:3, 0]

        else:
            data_module.prepare_data()
            data_module.setup()
            self.dictionary = data_module.trainset.D.t()  # torch.from_numpy()
            self.LUT_T1T2 = data_module.trainset.labels[:, 1:3, 0]

        self.dictionary = self.dictionary.cuda()
        self.LUT_T1T2 = self.LUT_T1T2.cuda()

    def FLOR_dictionary_based(self, batch, batch_idx, mode="img_test"):
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # Input variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)

        # Get input and matrix multiple
        # Ex (1, 1000) * (100, 1000)^T = 1*100
        # Ex (40, 1000) * (100, 1000)^T = 40*100
        x = torch.real(x)
        self.dictionary = torch.real(self.dictionary)
        simm = torch.matmul(x, self.dictionary)

        # Select index
        max_idx = torch.argmax(simm, dim=1)

        # Select T1T2 from LUT
        T1T2_hat = self.LUT_T1T2[max_idx]

        # Get loss
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")

        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": 0.0,
            },
            prog_bar=True,
            # sync_dist=True,
        )

        return T1T2_hat

    def validation_step(self, batch, batch_idx):
        self.FLOR_dictionary_based(batch["seqTe"], batch_idx, mode="val_seqTe")
        self.FLOR_dictionary_based(batch["phantomAll"], batch_idx, mode="val_phantomAll")
        self.FLOR_dictionary_based(batch["real"], batch_idx, mode="val_real")

    def test_step(self, batch, batch_idx):
        self.FLOR_dictionary_based(batch["seqTe"], batch_idx, mode="val_seqTe")
        self.FLOR_dictionary_based(batch["phantomAll"], batch_idx, mode="val_phantomAll")
        self.FLOR_dictionary_based(batch["real"], batch_idx, mode="val_real")


class LitModel_Test_not_working(pl.LightningModule):
    def __init__(
        self,
        exp_id,
        input_prod_size,
        enc_out_dim,
        latent_dim,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.step = self.step_en

        # Load saved model
        path = "/mnt/ssd/jj/Research/cauMedical/output/Split/" + str(exp_id) + "/last.ckpt"
        self.model = LitModel(input_prod_size, enc_out_dim, latent_dim)
        self.model.load_from_checkpoint(path)
        self.encoder = self.model.encoder.cuda()

    def step_en(self, batch, batch_idx, mode="img_test"):
        """
        x: (n, length)
        z: (n, length, 5) # should be
        """
        x, z = batch
        x = x.to(torch.float)
        z = z.to(torch.float)
        if z.shape[-1] > 10:
            z = z.transpose(1, 2)

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")
        self.log_dict(
            {
                f"{mode}_loss": loss.type(torch.float),
                f"{mode}_rec_loss": 0.0,
                f"{mode}_emb_loss": loss.type(torch.float),
            },
            prog_bar=True,
            # sync_dist=True,
        )
        if "test" in mode and batch_idx < 5:
            self.save_en(T1T2, T1T2_hat, batch_idx, mode)
        return loss

    def save_en(self, z, z_hat, batch_idx, mode):
        file_name = (
            "./output/Split/"
            + str(self.logger.version)
            + "/"
            + str(mode)
            + "_"
            + str(batch_idx)
            + "_saved_variables.npz"
        )
        np.savez(
            file_name,
            z_hat=z_hat.float().detach().cpu(),
            z=z.float().cpu(),
        )

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, mode="remove_testset_img_val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="remove_testset_img_test")  # remove_all_img_test
