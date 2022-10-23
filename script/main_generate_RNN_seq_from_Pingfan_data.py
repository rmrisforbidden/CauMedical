import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import relu
import numpy as np
from torch.autograd import Variable

import torch
import torch.nn.functional as F
import neptune.new as neptune

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
import os
import sys

from trainer import BlochSeqEqDecoder_our_RNN

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from module.Pingfan_data_seq_datamodule import MRIDataModule



def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--is_input_RF", type=int, default=0)
    

    parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitModel.add_model_specific_args(parser)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # data, model
    # ------------
    data_module_RNN = MRIDataModule(
            batch_size=args.batch_size, num_workers=4, test_type="seq", subsamp=5, seq_jump=5, is_input_RF=1,
            need_T1T2_logscale=True, need_TETR_second=True, need_RF_degree=True,
        )
    data_module_RNN.setup()
    test_loader = data_module_RNN.test_dataloader()

    # data_module_bloch = MRIDataModule(
    #         batch_size=args.batch_size, num_workers=4, test_type="seq", subsamp=5, seq_jump=5, is_input_RF=1,
    #         need_T1T2_logscale=False, need_TETR_second=False, need_RF_degree=False,
    #     )
    # data_module_RNN.setup()
    # test_loader2 = data_module_RNN.test_dataloader()
    # ------------
    # generate seq from RNN
    # ------------
    decoder = BlochSeqEqDecoder_our_RNN() #BlochSeqEqDecoder_RNN()

    # bloch = BlochSeqEqDecoder_ver2()
    seq = []
    pre_seq = []
    k=0
    for batch in test_loader:
        x, z = batch
        batch = (x, z.cuda())
        out = decoder.forward(batch)
        seq.append(out)
        pre_seq.append(x)
        if k>1:
            break
        k = k+1
    seq = torch.cat(seq, 0)
    pre_seq = torch.cat(pre_seq, 0)

    # Generate Bloch
    # for batch in test_loader2:
    #     x, zz = batch
    #     batch = (x, zz.cuda())
    #     outb = bloch.forward(batch)

    np.savez('./lightning_bolts/Bloch_decoder/data/approx/generte_our_RNN_seq_from_Pingfan_data.npz', 
            RNN=seq.detach().cpu().numpy(),
            Bloch=pre_seq.detach().cpu().numpy(),
            data_RNN = z.cpu().numpy(),
            ) #genBloch=outb.detach().cpu().numpy(), data_Bloch = zz.cpu().numpy()
    # np.savez('./lightning_bolts/Bloch_decoder/data/approx/generte_RNN_seq_from_Pingfan_data.npz', 
    #         RNN=seq,
    #         Bloch=pre_seq,
    #         genBloch=outb.detach().cpu().numpy())


if __name__ == "__main__":
    cli_main()
