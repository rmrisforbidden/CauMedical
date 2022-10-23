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

from trainer import LitModel, LitModel_Test, LitDictionary_baseline_Test

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# from module.Pingfan_data_seq_datamodule import MRIDataModule
from module.Pingfan_ver2_datamodule import MRIDataModule


class LitImgTester(LitModel):
    def __init__(
        self, input_prod_size, enc_out_dim, latent_dim, rec_lambda=1, decoder_type="our_RNN_bloch", is_emb_loss=1
    ):
        super().__init__(input_prod_size, enc_out_dim, latent_dim, rec_lambda, decoder_type, is_emb_loss=is_emb_loss)

    def test_step(self, batch, batch_idx):
        if "none" in self.decoder_type or "None" in self.decoder_type:
            self.step_en(batch, batch_idx, mode="test")
        elif "RNN" in self.decoder_type:
            self.step_en_de_RNN(batch, batch_idx, mode="test")
        else:
            self.step_en_de(batch, batch_idx, mode="test")
        return

    def step_en(self, batch, batch_idx, mode="test"):
        x, z = batch

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # Get rid of -inf in T1T2 (because of log10)
        T1T2[T1T2 == float("-inf")] = 0
        mask = torch.zeros_like(T1T2)
        mask[T1T2 > 0] = 1

        T1T2_hat = self.encoder.forward(x)
        loss = F.mse_loss(T1T2_hat * mask, T1T2, reduction="mean")
        self.log_dict(
            {
                f"{mode}_img_emb_loss": loss.type(torch.float),
            },
            prog_bar=True,
            sync_dist=True,
        )
        if mode == "test":
            self.save_en(x, T1T2, T1T2_hat)
        return loss

    def step_en_de_RNN(self, batch, batch_idx, mode="test"):
        x, z = batch

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # Get rid of -inf in T1T2 (because of log10)
        T1T2[T1T2 == float("-inf")] = 0
        mask = torch.zeros_like(T1T2)
        mask[T1T2 > 0] = 1

        # encoder
        T1T2_hat = self.encoder.forward(x)  # T1T2_hat: (n,2)
        T1T2_hat_ = T1T2_hat.unsqueeze(-1).repeat(1, 1, TE.shape[1]).transpose(1, 2)  # T1T2_hat: (n, length, 2)

        # decoder
        z_hat = torch.cat((RF, T1T2_hat_, TE, TR), 2)
        x_hat = self.decoder.forward((x, z_hat)).squeeze(-1)

        # loss
        rec_loss = F.mse_loss(mask * T1T2_hat, x, reduction="mean")
        emb_loss = F.mse_loss(T1T2_hat, T1T2, reduction="mean")

        self.log_dict(
            {
                f"{mode}_img_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            sync_dist=True,
        )
        if mode == "test":
            self.save_en_de(x, T1T2, x_hat, T1T2_hat)
        return emb_loss

    def step_en_de(self, batch, batch_idx, mode="test"):
        x, z = batch

        # variables
        T1T2 = z[:, 0, 1:3]  # T1T2: (n, 2)
        TE = z[:, :, 3].unsqueeze(-1)  # TE: (n, length, 1)
        TR = z[:, :, 2].unsqueeze(-1)  # TR: (n, length, 1)
        RF = z[:, :, 0].unsqueeze(-1)  # RF: (n, length, 1)

        # Get rid of -inf in T1T2 (because of log10)
        T1T2[T1T2 == float("-inf")] = 0
        mask = torch.zeros_like(T1T2)
        mask[T1T2 > 0] = 1

        T1T2_hat = self.encoder.forward(x)
        x_hat = self.decoder.forward(T1T2_hat)
        x_hat = x_hat.real
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        emb_loss = F.mse_loss(mask * T1T2_hat, T1T2, reduction="mean")

        self.log_dict(
            {
                f"{mode}_img_emb_loss": emb_loss.type(torch.float),
            },
            prog_bar=True,
            sync_dist=True,
        )
        if mode == "test":
            self.save_en_de(x, T1T2, x_hat, T1T2_hat)
        return emb_loss


def load_ckpt(exp_id, root_dir):
    path = os.path.join(root_dir, exp_id, "last.ckpt")
    return torch.load(path, map_location="cpu")


def set_prev_args(ckpt, args):
    # print(ckpt.keys(), '0000000000')
    # for k, v in ckpt.items():
    #     # if k == "data_dir":
    #     #     continue
    #     # if k == "exp_id":
    #     #     continue
    #     # if k == "default_root_dir":
    #     #     continue
    #     setattr(args, k, v)
    return args


def set_prev_args(ckpt, args):
    for k, v in ckpt["hyper_parameters"].items():
        if k == "data_dir":
            continue
        if k == "exp_id":
            continue
        if k == "default_root_dir":
            continue
        if k == "mratio":
            v = 1.0
        setattr(args, k, v)
    return args


def safe_model_loader(model, ckpt):
    try:
        model.load_state_dict(ckpt["state_dict"])
    except:
        ckpt["state_dict"]["model.fc2.weight"] = ckpt["state_dict"]["model.fc.weight"]
        ckpt["state_dict"]["model.fc2.bias"] = ckpt["state_dict"]["model.fc.bias"]
        del ckpt["state_dict"]["model.fc.weight"], ckpt["state_dict"]["model.fc.bias"]

        model.load_state_dict(ckpt["state_dict"])

    return


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--exp_id", default="", type=str)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train_data_type", type=str, default="seqTr")
    parser.add_argument("--test_data_type", type=str, default="real")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--is_dictionary_based", type=int, default=0)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # args : load saved args
    # ckpt = load_ckpt(args.exp_id, args.default_root_dir)
    # args = set_prev_args(ckpt, args)

    # ------------
    # data, model
    # ------------
    datamodule = None
    data_module = MRIDataModule(
        batch_size=args.batch_size,
        num_workers=4,
        train_data_type=args.train_data_type,
        test_data_type=args.test_data_type,
        # is_split_range_T1T2=True,
        subsamp=1,
        seq_jump=5,
        is_input_RF=1,
        need_T1T2_logscale=True,
        need_TETR_second=True,
        need_RF_degree=True,
    )
    data_module2 = MRIDataModule(
        batch_size=args.batch_size,
        num_workers=4,
        train_data_type=args.train_data_type,
        test_data_type=args.test_data_type,
        # is_split_range_T1T2=True,
        subsamp=1,
        seq_jump=5,
        is_input_RF=1,
        need_T1T2_logscale=True,
        need_TETR_second=True,
        need_RF_degree=True,
    )

    # enc_out_dim = 300 # 512,
    # latent_dim = 2  # T1 and T2,
    # input_prod_size = 1000 #(1000//args.subsamp,)  # self.L//subsamp of bloch decoder, (200,)

    # model = LitModel(input_prod_size, enc_out_dim, latent_dim, rec_lambda=args.rec_lambda, decoder_type=args.decoder_type)
    # ------------
    # Trainer
    # ------------
    exp_id = None if args.is_dictionary_based > 0 else args.exp_id
    if True:
        API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="
        run = neptune.init(
            api_token=API_KEY,
            project=f"CausalMRI/{args.default_root_dir.split('/')[-1]}",
            capture_stdout=False,
            run=exp_id,
            # "sync"
        )  # mode="debug",
        neptune_logger = NeptuneLogger(run=run, log_model_checkpoints=False)

    # Load model
    # trainer = pl.Trainer.from_argparse_args(args, accelerator="gpu", gpus=1, logger=neptune_logger)
    trainer = pl.Trainer.from_argparse_args(
        args,
        progress_bar_refresh_rate=20,
        check_val_every_n_epoch=5,
        precision=16,
        logger=neptune_logger,
        accelerator="gpu",
    )
    # model = LitImgTester(**vars(args))
    if args.is_dictionary_based > 0:
        model = LitDictionary_baseline_Test(args.train_data_type, args.test_data_type, data_module2)
    else:
        enc_out_dim = 300  # 512,
        latent_dim = 2  # T1 and T2,
        input_prod_size = 1000  # (1000//args.subsamp,)  # self.L//subsamp of bloch decoder, (200,)
        model = LitModel_Test(
            args.exp_id,
            input_prod_size,
            enc_out_dim,
            latent_dim,
            args.test_data_type,
        )
    # safe_model_loader(model, ckpt)

    # test
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
