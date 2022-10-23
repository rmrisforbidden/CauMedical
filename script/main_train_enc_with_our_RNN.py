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

from trainer import LitModel

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# from module.Pingfan_data_seq_datamodule import MRIDataModule
from module.Pingfan_ver2_datamodule import MRIDataModule


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--approx_model_type", type=str, default="RNN")
    parser.add_argument("--is_input_RF", type=int, default=1)
    parser.add_argument("--decoder_type", type=str, default="our_RNN")
    parser.add_argument("--encoder_type", type=str, default="FC")
    parser.add_argument("--rec_lambda", type=float, default=1.0)
    parser.add_argument("--is_emb_loss", type=int, default=1)
    parser.add_argument("--train_data_type", type=str, default="seqTr")
    parser.add_argument("--test_data_type", type=str, default="real")

    parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitModel.add_model_specific_args(parser)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # data, model
    # ------------
    need_T1T2_logscale = True #False if "causal" in args.decoder_type else True
    need_TETR_second = True #False if "causal" in args.decoder_type else True
    need_RF_degree = True #False if "causal" in args.decoder_type else True

    data_module = MRIDataModule(
        batch_size=args.batch_size,
        num_workers=4,
        train_data_type=args.train_data_type,
        test_data_type=args.test_data_type,
        # is_split_range_T1T2=True,
        subsamp=1,
        seq_jump=5,
        is_input_RF=1,
        need_T1T2_logscale=need_T1T2_logscale,
        need_TETR_second=need_TETR_second,
        need_RF_degree=need_RF_degree,
    )

    enc_out_dim = 300  # 512,
    latent_dim = 2  # T1 and T2,
    input_prod_size = 1000  # (1000//args.subsamp,)  # self.L//subsamp of bloch decoder, (200,)
    model = LitModel(
        input_prod_size,
        enc_out_dim,
        latent_dim,
        rec_lambda=args.rec_lambda,
        decoder_type=args.decoder_type,
        encoder_type=args.encoder_type,
        is_emb_loss=args.is_emb_loss,
    )
    # ------------
    # Trainer
    # ------------
    if True:
        API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="
        run = neptune.init(
            api_token=API_KEY,
            project=f"CausalMRI/{args.default_root_dir.split('/')[-1]}",
            capture_stdout=False,
            # "sync"
        )  # mode="debug",
        neptune_logger = NeptuneLogger(run=run)
        dirpath = os.path.join(args.default_root_dir, neptune_logger.version)
        neptune_logger.log_hyperparams(args)

        # ------------
        # callbacks
        # ------------
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val_seqTe_loss",
            filename="checkpt-{epoch:02d}-{val_loss:.2f}",
            # filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer.from_argparse_args(
            args,
            progress_bar_refresh_rate=20,
            check_val_every_n_epoch=5,
            precision=16,
            logger=neptune_logger,
            callbacks=[checkpoint_callback, lr_monitor],
        )

    # train
    trainer.fit(model, datamodule=data_module)

    # test
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
