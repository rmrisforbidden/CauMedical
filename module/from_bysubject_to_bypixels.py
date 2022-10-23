import scipy.io
import sys
import h5py
import numpy as np
import mat73
import torch
import matplotlib.pyplot as plt

D = {}  # D : (sub, slice, 128*128, 1000)
labels = {}  # labels : (sub, slice, 128*128, (T1, T2)) = ()


subject_list = ["04", "05", "06", "18", "20", "38", "41", "42", "43", "44"]
n_slice_per_subject = 12
L = 1000
subsamp = 1
len_seq = L // subsamp
need_T1T2_logscale = True
is_split_range_T1T2 = False
T1_condition_threshold = 1000
T2_condition_threshold = 50
need_RF_degree = True
need_TETR_second = True
mode = "train"
slice_num_now = 0
pixel_num_now = 0


for i in range(len(subject_list)):
    file = "subject" + str(subject_list[i]) + "_MRIs.mat"
    file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_subject/" + file
    data_dict = mat73.loadmat(file_name)

    for s in range(n_slice_per_subject):

        if s > len(data_dict["X_all"]) - 1:  # some subjects only have 10 slices instead of 12 slices
            continue

        # X_all(fully sampled image sequence)
        D_slice = torch.from_numpy(data_dict["X_all"][s][:, 0:L:subsamp])  # (128, 128, 1000)
        D_slice = torch.flatten(D_slice, 0, 1)  # (128*128, 1000)
        D_slice = torch.nn.functional.normalize(D_slice, p=2.0, dim=1)

        # LUT (T1,T2,PD)
        T1 = torch.from_numpy(data_dict["LUT"][0][s]).unsqueeze(-1)  # (128, 128, 1)
        T2 = torch.from_numpy(data_dict["LUT"][1][s]).unsqueeze(-1)  # (128, 128, 1)
        labels_slice = torch.cat((T1, T2), -1)  # (128, 128, 2)
        labels_slice = torch.flatten(labels_slice, 0, 1)  # (128*128, 2)

        if is_split_range_T1T2:
            # Split train and test set : T1 0~2500, 2500~5000
            T1 = labels_slice[:, 0]
            T2 = labels_slice[:, 1]
            condition = (
                (T1 < T1_condition_threshold) & (T2 < T2_condition_threshold)
                if "train" in mode
                else (T1 > T1_condition_threshold) & (T2 > T2_condition_threshold)
            )
            labels_slice = labels_slice[condition]  # (condition_num, 1, 1000)
            D_slice = D_slice[condition]  # (condition_num, 1, 1000)
            print(labels_slice.shape, D_slice.shape)

        # T1T2 scale
        T1 = labels_slice[:, 0].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # (128*128, 1, 1000)
        T2 = labels_slice[:, 1].unsqueeze(1).repeat(1, len_seq).unsqueeze(1)  # (128*128, 1, 1000)
        if need_T1T2_logscale == True:
            T1 = torch.log10(T1)
            T2 = torch.log10(T2)

        # RFpulses : (128*128, 1, 1000)
        num_seq = len(labels_slice)
        RFpulses = torch.from_numpy(data_dict["Params"]["RFpulses"][0:L:subsamp])  # (1000,)
        RFpulses = RFpulses.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

        # TR : (128*128, 1, 1000)
        TR = torch.from_numpy(data_dict["Params"]["TR"][0:L:subsamp])  # (1000, )
        TR = TR.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

        # TE : (128*128, 1, 1000)
        TE = torch.ones(num_seq, 1, L // subsamp) * data_dict["Params"]["RFpulses"]  # (128*128, 1, 1000)

        # RFpulses, TR, TE scale
        if need_RF_degree == True:
            RFpulses = RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            TR = TR / 1000
            TE = TE / 1000

        # Concat
        labels_slice = torch.cat((RFpulses, T1, T2, TE, TR), 1)  # (condition, 5, 1000)

        # Save by pixel
        for p in range(128 * 128):

            sl_file = "pixel" + str(pixel_num_now) + "_MRIs"
            sl_file_name = (
                "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_pixel/" + sl_file
            )
            np.savez(sl_file_name, X_all=D_slice, labels=labels_slice)

            # Update pixel_num_now
            pixel_num_now += 1

        # Update slice_num_now
        print("slice: ", slice_num_now, "pixel: ", pixel_num_now)
        slice_num_now += 1
