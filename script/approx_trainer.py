# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.nn.functional import relu
# import numpy as np
# from torch.autograd import Variable

# import neptune.new as neptune
# import pytorch_lightning as pl
# from pytorch_lightning.loggers.neptune import NeptuneLogger
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from argparse import ArgumentParser
# import os
# import sys

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
# from Bloch_decoder.top_MRI_simulator import Top_MRI_simulator
# from Bloch_decoder.utils.generate_RF_TR import generate_RF_TR

# import torch.nn as nn

# class GRU(nn.Module):
#     def __init__(self,input_size=5, hidden_size_1=32, hidden_size_2=32, hidden_size_3=32, output_size=3, num_layers=1, device='cuda'):
#         super(GRU, self).__init__()
#         """
#         # This model is trained in the paper: 
#         # output : (m, dm/dT1, dm/dT2)
#         # input : (RF, T1, T2, TE, TR)
#         """
#         self.input_size = input_size
#         self.hidden_size_1 = hidden_size_1
#         self.hidden_size_2 = hidden_size_2
#         self.hidden_size_3 = hidden_size_3
#         self.num_layers = num_layers
#         self.device = device
        
#         self.gru_1 = nn.GRU(input_size, hidden_size_1, num_layers, batch_first=True)
#         self.gru_2 = nn.GRU(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
#         self.gru_3 = nn.GRU(hidden_size_1, hidden_size_3, num_layers, batch_first=True)
#         self.fc_out = nn.Linear(hidden_size_3, output_size)

#     def forward(self, x, h_set):
#         input_X = x
#         out_gru_1 ,h_set[0] = self.gru_1(input_X, h_set[0])   # h_1
#         out_gru_2 ,h_set[1] = self.gru_2(out_gru_1, h_set[1]) # h_2
#         out_gru_3 ,h_set[2] = self.gru_2(out_gru_2, h_set[2]) # h_3
#         out_Dense_out = self.fc_out(out_gru_3) 

#         return out_Dense_out, h_set
    
#     def init_hidden_set(self):
#         h_set = {}
#         h_set[0] = torch.zeros(self.num_layers, self.hidden_size_1, device=self.device) # h_1
#         h_set[1] = torch.zeros(self.num_layers, self.hidden_size_2, device=self.device) # h_2
#         h_set[2] = torch.zeros(self.num_layers, self.hidden_size_3, device=self.device) # h_3
#         return h_set


# class FC_Decoder(nn.Module):
#     def __init__(
#         self, input_prod_size=200, enc_out_dim=300, latent_dim=2
#     ): 
#         super(FC_Decoder, self).__init__()
#         self.linear_ = nn.Linear(latent_dim, enc_out_dim)  # .to(torch.complex64)
#         self.fc4 = nn.Linear(enc_out_dim, enc_out_dim)  # .to(torch.complex64)
#         self.fc5 = nn.Linear(enc_out_dim, input_prod_size)  # .to(torch.complex64)
#         self.fc6 = nn.Linear(input_prod_size, input_prod_size)  # .to(torch.complex64)

#     def forward(self, z):
#         print(z.shape)
#         print('--'*30)
#         h4 = F.relu(self.linear_(z)) #F.softplus(self.fc1(x), beta=1)  # real_imaginary_relu(self.fc1(x))  # F.softplus(self.fc1(x), beta=1)  #
#         h5 = F.relu(self.fc4(h4)) #F.softplus(self.fc1(h1), beta=1)
#         h6 = F.relu(self.fc5(h5)) #F.softplus(self.fc1(h1), beta=1)  
#         return self.fc6(h6)  # torch.sigmoid(self.fc4(h3))

# class BlochSeqEqDecoder:
#     def __init__(self, L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
#         self.L = L // subsamp
#         self.TE = TE

#         # Real value
#         RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
#         RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
#         RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
#         TR = TR[0:L:subsamp]

#         self.TR = TR
#         self.RFpulses = RFpulses
        
#     def forward(self, decoder_input):
#         # Decoder_input : (n,2)
#         T1_values = decoder_input[:, 0]
#         T2_values = decoder_input[:, 1]
#         simulator = Top_MRI_simulator()
#         D, LUT = simulator.build_dictionary_fisp_seq(
#             T1_values=T1_values, T2_values=T2_values, L=self.L, TE=self.TE, RFpulses=self.RFpulses, TR=self.TR
#         )
#         return D

# class BlochSeqEqDecoder_RNN:
#     def __init__(self, L=1000, subsamp=5, TE=10):  # larger subsamp makes shorter length
#         self.L = L // subsamp
#         self.TE = TE

#         # Real value
#         RFpulses, TR = generate_RF_TR(L)  #%Load slowly changing RF and TR values
#         RFpulses = RFpulses[0:L:subsamp]  #% undersampling in time dimension.
#         RFpulses = RFpulses * 1j  # %to avoid complex values of X and D
#         TR = TR[0:L:subsamp]

#         self.TR = TR
#         self.RFpulses = RFpulses

#         simulator = GRU()
#         simulator.load_state_dict(torch.load('new_pytorch_RNN_EPG.pt'))
#         simulator.eval()
        
#     def forward(self, decoder_input):
#         # Decoder_input : (n,5)
#         seq=[]
#         h_set = model.init_hidden_set()
#         for i in range(input_.shape[1]):
#             inp = input_[:,:,i]
#             out, h_set = nm.forward(inp, h_set)
#             seq.append(out[:,0])
#         m, dm_dT1, dm_dT2 = simulator(decoder_input)
#         simulator.forward(x, h_set)
#         return m

# class LitTrainApproxModel(pl.LightningModule):
#     def __init__(self, model_type='RNN', is_input_RF=0):
#         super().__init__()
#         input_size = 5 if is_input_RF>0 else 2
#         if 'RNN' in model_type: 
#             self.model = GRU(input_size=input_size)
#         else:
#             self.model = FC_Decoder(latent_dim=input_size)
        
#     def step(self, batch, batch_idx, mode='test'):
#         x, z = batch
#         x_hat = self.model.forward(z)
#         loss = F.mse_loss(x_hat, x, reduction="mean")
#         self.log_dict(
#             {
#                 f"{mode}_loss": loss.type(torch.float),
#             },
#             prog_bar=True,
#             sync_dist=True,
#         )
#         return loss

#     def training_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx, mode='train')
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.step(batch, batch_idx, mode='val')

#     def test_step(self, batch, batch_idx):
#         self.step(batch, batch_idx, mode='test')

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

# class LitUseApproxModel(pl.LightningModule):
#     def __init__(self, model_type='RNN', is_input_RF=1):
#         super().__init__()
#         input_size = 5 if is_input_RF>0 else 2
#         if 'RNN' in model_type: 
#             self.model = GRU(input_size=input_size)
#         else:
#             self.model = FC_Decoder(latent_dim=input_size)
        
#     def step(self, batch, batch_idx, mode='test'):
#         x, z = batch
#         x_hat = self.model.forward(z)
#         loss = F.mse_loss(x_hat, x, reduction="mean")
#         self.log_dict(
#             {
#                 f"{mode}_loss": loss.type(torch.float),
#             },
#             prog_bar=True,
#             sync_dist=True,
#         )
#         return loss

#     def training_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx, mode='train')
#         return loss

#     def validation_step(self, batch, batch_idx):
#         self.step(batch, batch_idx, mode='val')

#     def test_step(self, batch, batch_idx):
#         self.step(batch, batch_idx, mode='test')

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-3)




# """
# class GRU(nn.Module):
#     def __init__(self,input_size=5, hidden_size_1=32, hidden_size_2=32, hidden_size_3=200, output_size=200, num_layers=1, device='cuda'):
#         super(GRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size_1 = hidden_size_1
#         self.hidden_size_2 = hidden_size_2
#         self.hidden_size_3 = hidden_size_3
#         self.num_layers = num_layers
#         self.device = device
        
#         self.gru_1 = nn.GRU(input_size, hidden_size_1, num_layers, batch_first=True)
#         self.gru_2 = nn.GRU(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
#         self.gru_3 = nn.GRU(hidden_size_2, hidden_size_3, num_layers, batch_first=True)
#         self.fc_out = nn.Linear(hidden_size_3, output_size)

#     def forward(self, x, h_1, h_2, h_3):
#         input_X = x
#         out_gru_1 , h_1 = self.gru_1(input_X, h_1)
#         out_gru_2 , h_2 = self.gru_2(out_gru_1, h_2)
#         out_gru_3 , h_3 = self.gru_3(out_gru_2, h_3)
#         out_Dense_out = self.fc_out(out_gru_3)
#         return out_Dense_out, h_1, h_2, h_3

#     def hidden_init():
#         h_1 = torch.zeros(self.num_layers, 1, self.hidden_size_1, device=self.device)
#         h_2 = torch.zeros(self.num_layers, 1, self.hidden_size_2, device=self.device)
#         h_3 = torch.zeros(self.num_layers, 1, self.hidden_size_3, device=self.device)
#         # h_2 = torch.zeros(self.num_layers, input_X.size(0), self.hidden_size_2, device=self.device)

# # previous
# class GRU(nn.Module):
#     def __init__(self,input_size=2, hidden_size_1=32, hidden_size_2=32, hidden_size_3=200, output_size=200, num_layers=1, device='cuda'):
#         super(GRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size_1 = hidden_size_1
#         self.hidden_size_2 = hidden_size_2
#         self.hidden_size_3 = hidden_size_3
#         self.num_layers = num_layers
#         self.device = device
        
#         self.gru_1 = nn.GRU(input_size, hidden_size_1, num_layers, batch_first=True)
#         self.gru_2 = nn.GRU(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
#         self.gru_3 = nn.GRU(hidden_size_2, hidden_size_3, num_layers, batch_first=True)
#         self.fc_1 = nn.Linear(hidden_size_3, hidden_size_3)
#         self.fc_out = nn.Linear(hidden_size_3, output_size)

#     def forward(self, x):
#         input_X = x
#         h_1 = torch.zeros(self.num_layers, self.hidden_size_1, device=self.device)
#         h_2 = torch.zeros(self.num_layers, self.hidden_size_2, device=self.device)
#         h_3 = torch.zeros(self.num_layers, self.hidden_size_3, device=self.device)
#         # h_2 = torch.zeros(self.num_layers, input_X.size(0), self.hidden_size_2, device=self.device)
#         # h_3 = torch.zeros(self.num_layers, input_X.size(0), self.hidden_size_3, device=self.device)

#         out_gru_1 , h_1 = self.gru_1(input_X, h_1)
#         out_gru_2 , h_2 = self.gru_2(out_gru_1, h_2)
#         out_gru_3 , h_3 = self.gru_3(out_gru_2, h_3)
#         out_Dense_1 = self.fc_1(out_gru_3) 
#         out_Dense_out = self.fc_out(out_Dense_1)
#         return out_Dense_out
# """