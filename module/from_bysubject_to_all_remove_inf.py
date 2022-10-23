from sqlite3 import enable_shared_cache
import scipy.io
import sys
import h5py
import numpy as np
import mat73
import torch
import matplotlib.pyplot as plt

D = []  # D : (sub, slice, 128*128, 1000)
labels = []  # labels : (sub, slice, 128*128, (T1, T2)) = ()


subject_list = ["04", "05", "06", "18", "20", "38", "41", "42", "43", "44"]
n_slice_per_subject = 10
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

################################
is_two_slices = True
start_range = 4 if is_two_slices else 0
end_range = 6 if is_two_slices else n_slice_per_subject
slice_num_now_tr = 16
slice_num_now_all = 19

is_one_slices = False
start_range = 5 if is_one_slices else 0
end_range = 6 if is_one_slices else n_slice_per_subject
slice_num_now_tr = 7
slice_num_now_all = 9

is_save_np = True
################################


for i in range(len(subject_list)):
    file = "subject" + str(subject_list[i]) + "_MRIs.mat"
    file_name = "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/image_by_subject/" + file
    data_dict = mat73.loadmat(file_name)

    for s in range(start_range, end_range):

        if s > len(data_dict["X_all"]) - 1:  # some subjects only have 10 slices instead of 12 slices
            continue

        # X_all(fully sampled image sequence)
        D_slice = torch.from_numpy(data_dict["X_all"][s][:, 0:L:subsamp])  # (128, 128, 1000)
        D_slice = torch.flatten(D_slice, 0, 1)  # (128*128, 1000)
        D_slice = np.real(D_slice)
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

            # Replace -inf with -100
            T1_log = torch.nan_to_num(torch.log10(T1), neginf=-100)
            T2_log = torch.nan_to_num(torch.log10(T2), neginf=-100)

            # Get index that the value is -100
            T1_idx_n100 = torch.where(T1_log != -100)[0]
            T2_idx_n100 = torch.where(T2_log != -100)[0]
            T1T2_idx_n100 = np.intersect1d(T1_idx_n100, T2_idx_n100)

            # Get rid of -100
            D_slice = D_slice[T1T2_idx_n100]
            T1 = T1_log[T1T2_idx_n100]
            T2 = T2_log[T1T2_idx_n100]

        # RFpulses : (128*128, 1, 1000) : Use only imaginary part, as real parts are all zeros.
        num_seq = len(T1)
        RFpulses = data_dict["Params"]["RFpulses"][0:L:subsamp]  # (1000,)
        is_RFpulses_complex = np.iscomplex(RFpulses).sum() > 0
        if is_RFpulses_complex:
            RFpulses = torch.from_numpy(np.imag(RFpulses))
        RFpulses = RFpulses.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

        # TR : (128*128, 1, 1000)
        TR = torch.from_numpy(data_dict["Params"]["TR"][0:L:subsamp])  # (1000, )
        TR = TR.repeat(num_seq, 1).unsqueeze(1)  # (128*128, 1, 1000)

        # TE : (128*128, 1, 1000)
        TE = torch.ones(num_seq, 1, L // subsamp) * data_dict["Params"]["TE"]  # (128*128, 1, 1000)

        # RFpulses, TR, TE scale
        if need_RF_degree == True:
            RFpulses = RFpulses * 180 / torch.pi

        if need_TETR_second == True:
            TR = TR / 1000
            TE = TE / 1000

        # Concat
        labels_slice = torch.cat((RFpulses, T1, T2, TE, TR), 1)  # (condition, 5, 1000)

        # Concat for all
        if len(D) < 1:
            D = D_slice
            labels = labels_slice
        else:
            D = torch.cat((D, D_slice), 0)
            labels = torch.cat((labels, labels_slice), 0)
        print(slice_num_now, D.shape, labels.shape)

        # Save
        if is_two_slices or is_one_slices:
            if slice_num_now == slice_num_now_tr:
                sl_file = "train_10subject_4_5_slices_MRIs_new"  # "train_10subject_5_slices_MRIs"  # "train_10subject_4_5_slices_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )
                if is_save_np:
                    np.savez(sl_file_name + ".npz", X_all=D, labels=labels)
                else:
                    hf = h5py.File(sl_file_name + ".h5", "w")
                    hf.create_dataset("X_all", data=D, compression="gzip", chunks=True)
                    hf.create_dataset("labels", data=labels, compression="gzip", chunks=True)
                    hf.close()

                all_D = D
                D = []
                all_labels = labels
                labels = []
                print(sl_file)

            if slice_num_now == slice_num_now_all:
                sl_file = "test_10subject_4_5_slices_MRIs_new"  # "test_10subject_5_slices_MRIs"  # "test_10subject_4_5_slices_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )
                if is_save_np:
                    np.savez(sl_file_name + ".npz", X_all=D, labels=labels)
                else:
                    hf = h5py.File(sl_file_name + ".h5", "w")
                    hf.create_dataset("X_all", data=D, compression="gzip", chunks=True)
                    hf.create_dataset("labels", data=labels, compression="gzip", chunks=True)
                    hf.close()
                print(sl_file)

            if slice_num_now == slice_num_now_all:
                sl_file = "all_10subject_4_5_slices_MRIs_new"  # "all_10subject_5_slices_MRIs"  # "all_10subject_4_5_slices_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )
                save_all_D = torch.cat((all_D, D), 0)
                save_all_labels = torch.cat((all_labels, labels), 0)

                if is_save_np:
                    np.savez(sl_file_name + ".npz", X_all=save_all_D, labels=save_all_labels)
                else:
                    hf = h5py.File(sl_file_name + ".h5", "w")
                    hf.create_dataset("X_all", data=save_all_D, compression="gzip", chunks=True)
                    hf.create_dataset("labels", data=save_all_labels, compression="gzip", chunks=True)
                    hf.close()
                print(sl_file)
                print(done)
        else:
            if slice_num_now == 69:
                sl_file = "train_all_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )

                hf = h5py.File(sl_file_name + ".h5", "w")
                hf.create_dataset("X_all", data=D, compression="gzip", chunks=True)
                hf.create_dataset("labels", data=labels, compression="gzip", chunks=True)
                hf.close()
                all_D = D
                D = []
                all_labels = labels
                labels = []
                print(sl_file)

            if slice_num_now == 89:  # 11:
                sl_file = "test_all_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )

                hf = h5py.File(sl_file_name + ".h5", "w")
                hf.create_dataset("X_all", data=D, compression="gzip", chunks=True)
                hf.create_dataset("labels", data=labels, compression="gzip", chunks=True)
                hf.close()
                print(sl_file)

            if slice_num_now == 89:
                sl_file = "all_MRIs"  # + str(slice_num_now) + "_MRIs"
                sl_file_name = (
                    "/mnt/ssd/jj/Research/cauMedical/lightning_bolts/Bloch_decoder/data/Pingfan/remove_image_all/"
                    + sl_file
                )
                save_all_D = torch.cat((all_D, D), 0)
                save_all_labels = torch.cat((all_labels, labels), 0)
                hf = h5py.File(sl_file_name + ".h5", "w")
                hf.create_dataset("X_all", data=save_all_D, compression="gzip", chunks=True)
                hf.create_dataset("labels", data=save_all_labels, compression="gzip", chunks=True)
                hf.close()
                print(sl_file)

        # Update slice_num
        slice_num_now += 1
