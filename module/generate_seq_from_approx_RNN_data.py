import pwd
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import os
import scipy.io as sio
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Bloch_decoder.All_simulator import Top_MRI_simulator


# parameters setup
nTrain = 1000 
nTotal = 1500 
nOffset = 0
nSeq_Length = 224 * 5 # inversion pulse as the first TR
nInput =  5   # theta, T1, T2, TE, TR
nOutput = 3   # m, dm/dT1, dm/dT2
nBatch = 200
nRNN_Unit = 32
nEpoch = 1500
nVal_BatchSize = 6  # Number of batches used for validation during training
# Initial
Inv = []
# In
RFtrain=[]
T1=[]
T2=[]
TE = []
TR = []
# Out
Dictionary = []

#########################################################################
# model
class BlochSeqEqDecoder:
    def __init__(self):  # larger subsamp makes shorter length
        pass
        
    def forward(self, T1, T2, TR, RF, TE):
        # Decoder_input : (n,2)
        L = len(TR[0])
        T1_values = T1
        T2_values = T2
        simulator = Top_MRI_simulator()
        D, LUT = simulator.build_dictionary_fisp_seq(
            T1_values=T1_values, T2_values=T2_values, L=L, TE=TE, RFpulses=RF, TR=TR
        )
        return D
model=BlochSeqEqDecoder()
#########################################################################

#########################################################################
# generate sequence and save
def generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name):
    # cat
    TR = torch.from_numpy(np.concatenate(TR, 0))
    RFtrain = torch.from_numpy(np.concatenate(RFtrain, 0))
    TE = torch.from_numpy(np.concatenate(TE, 0))
    T1 = torch.from_numpy(np.expand_dims(np.concatenate(T1, 0),1))
    T2 = torch.from_numpy(np.expand_dims(np.concatenate(T2, 0),1))

    # Scale change
    TR = TR*1000
    TE = TE*1000
    T1 = torch.pow(10, T1)
    T2 = torch.pow(10, T2)

    # generate sequence
    model=BlochSeqEqDecoder()
    # T1=T1[:5]
    # T2=T2[:5]
    # TE=TE[:3]
    # TR=TR[:3]
    # RFtrain=RFtrain[:3]
    # TE=TE[:3]
    D = model.forward(T1=T1, T2=T2, TR=TR, RF=RFtrain, TE=TE)
    
    # save
    print(save_file_name)
    save={}
    save['TR'] = TR
    save['T1'] = T1
    save['T2'] = T2
    save['TE'] = TE
    save['RF'] = RFtrain
    save['generated_sequence'] = D
    np.save(save_file_name, save)
#########################################################################

filename = ["1Spline5"]

for ii in range(len(filename)):
    """
    Data shape info
    # TE = (3000, 1120)
    # TR = (3000, 1120)
    # RF = (3000, 1120)
    # T1T2 = (3000, 2)  T1T2=np.concatenate((np.expand_dims(T1T2B1[:,0],1), np.expand_dims(T1T2B1[:,1],1)),1)
    """
    RFtrain=[]
    T1=[]
    T2=[]
    TE = []
    TR = []
    Inv = []
    # read file one
    str1 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"  + filename[ii] + "Flex.mat"
    traindata = sio.loadmat(str1)
    RFtrain.append(traindata.get("rf")) # flipangles

    # read file two
    str2 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/Gauss/"  + filename[ii] + ".mat"
    traindata = sio.loadmat(str2)
    temp0 = traindata.get("dictionary")
    Dictionary.append(temp0)

    TE.append(traindata.get("TEAll"))
    TR.append(traindata.get("TRAll"))

    T1T2B1 = traindata.get("T1T2B1")
    T1.append(T1T2B1[:,0])
    T2.append(T1T2B1[:,1])
    Inv.append(-180.0 * np.ones(shape=(1))) # Changed on May,13th

    #########################################################################
    # generate sequence and save
    save_file_name = 'lightning_bolts/Bloch_decoder/data/approx/'+'new_part1_' + str(filename[ii])+'.npy'
    generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name)
    #########################################################################

"""
# 2) Load data
# PART ONE (INVERSE)
# data selection
# filename = ["1Spline5","2Spline11","3SinSquared5","4SplineNoise11","5PieceConstant5"] 
filename = ["1Spline5","2Spline11","3SinSquared5","4SplineNoise11","5PieceConstant5"]

for ii in range(len(filename)):

    # Data shape info
    # TE = (3000, 1120)
    # TR = (3000, 1120)
    # RF = (3000, 1120)
    # T1T2 = (3000, 2)  T1T2=np.concatenate((np.expand_dims(T1T2B1[:,0],1), np.expand_dims(T1T2B1[:,1],1)),1)

    RFtrain=[]
    T1=[]
    T2=[]
    TE = []
    TR = []
    Inv = []
    # read file one
    str1 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"  + filename[ii] + "Flex.mat"
    traindata = sio.loadmat(str1)
    RFtrain.append(traindata.get("rf")) # flipangles

    # read file two
    str2 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/Gauss/"  + filename[ii] + ".mat"
    traindata = sio.loadmat(str2)
    temp0 = traindata.get("dictionary")
    Dictionary.append(temp0)

    TE.append(traindata.get("TEAll"))
    TR.append(traindata.get("TRAll"))

    T1T2B1 = traindata.get("T1T2B1")
    T1.append(T1T2B1[:,0])
    T2.append(T1T2B1[:,1])
    Inv.append(-180.0 * np.ones(shape=(1))) # Changed on May,13th

    #########################################################################
    # generate sequence and save
    save_file_name = 'lightning_bolts/Bloch_decoder/data/approx/'+'part1_' + str(filename[ii])+'.npy'
    generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name)
    #########################################################################


# PART TWO (NO INVERSE)
# data selection
RFtrain=[]
T1=[]
T2=[]
TE = []
TR = []
Inv = []
filename = ["1Spline5","2Spline11","3SinSquared5","4SplineNoise11","5PieceConstant5"] 
for ii in range(len(filename)):
    # read file one
    str1 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"  + filename[ii] + "Flex.mat"
    traindata = sio.loadmat(str1)
    RFtrain.append(traindata.get("rf")) # flipangles

    # read file two
    str2 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/Gauss_NoInv/"  + filename[ii] + ".mat"
    traindata = sio.loadmat(str2)
    temp0 = traindata.get("dictionary")
    # temp0 = np.swapaxes(temp0,0,2)
    Dictionary.append(temp0)

    TE.append(traindata.get("TEAll"))
    TR.append(traindata.get("TRAll"))

    T1T2B1 = traindata.get("T1T2B1")
    T1.append(T1T2B1[:,0])
    T2.append(T1T2B1[:,1])
    # Inv.append(0.0 * np.ones(shape=(1)))
    Inv.append(180.0 * np.ones(shape=(1))) # Changed on May13th,2020

    #########################################################################
    # generate sequence and save
    save_file_name = 'lightning_bolts/Bloch_decoder/data/approx/'+'part2_' + str(filename[ii])+'.npy'
    generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name)
    #########################################################################

# PART THREE (INVERSE, FLEX TETR)
# data selection
filename = ["1Spline5","2Spline11","3SinSquared5","4SplineNoise11","5PieceConstant5"] 
for ii in range(len(filename)):
    # read file one
    str1 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"  + filename[ii] + "Flex.mat"
    traindata = sio.loadmat(str1)
    RFtrain.append(traindata.get("rf")) # flipangles

    # read file two
    str2 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/Gauss/"  + filename[ii] + "_FlexTETR.mat"
    traindata = sio.loadmat(str2)
    temp0 = traindata.get("dictionary")
    # temp0 = np.swapaxes(temp0,0,2)
    Dictionary.append(temp0)

    TE.append(traindata.get("TEAll"))
    TR.append(traindata.get("TRAll"))

    T1T2B1 = traindata.get("T1T2B1")
    T1.append(T1T2B1[:,0])
    T2.append(T1T2B1[:,1])
    Inv.append(-180.0 * np.ones(shape=(1))) # Changed on May,13th
    # Inv.append(180.0 * np.ones(shape=(1))) 

    #########################################################################
    # generate sequence and save
    save_file_name = 'lightning_bolts/Bloch_decoder/data/approx/'+'part3_' + str(filename[ii])+'.npy'
    generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name)
    #########################################################################

# PART FOUR (NO INVERSE,FLEX TETR)
# data selection
filename = ["1Spline5","2Spline11","3SinSquared5","4SplineNoise11","5PieceConstant5"] 
for ii in range(len(filename)):
    # read file one
    str1 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/"  + filename[ii] + "Flex.mat"
    traindata = sio.loadmat(str1)
    RFtrain.append(traindata.get("rf")) # flipangles

    # read file two
    str2 = "lightning_bolts/Bloch_decoder/data/Experiments/TrainingData/Gauss_NoInv/"  + filename[ii] + "_FlexTETR.mat"
    traindata = sio.loadmat(str2)
    temp0 = traindata.get("dictionary")
    # temp0 = np.swapaxes(temp0,0,2)
    Dictionary.append(temp0)

    TE.append(traindata.get("TEAll"))
    TR.append(traindata.get("TRAll"))

    T1T2B1 = traindata.get("T1T2B1")
    T1.append(T1T2B1[:,0])
    T2.append(T1T2B1[:,1])
    # Inv.append(0.0 * np.ones(shape=(1)))
    Inv.append(180.0 * np.ones(shape=(1))) # Changed on May13th,2020

    #########################################################################
    # generate sequence and save
    save_file_name = 'lightning_bolts/Bloch_decoder/data/approx/'+'part4_' + str(filename[ii])+'.npy'
    generate_sequence_f(TR, RFtrain, TE, T1, T2, save_file_name)
    #########################################################################
"""







