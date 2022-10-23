import torch
import torch.nn.functional as F
import neptune.new as neptune

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from pl_bolts.models.autoencoders import AE
from module.datamodule import CIFAR10DataModule
from module.brainweb_datamodule import BrainWebDataModule
from pl_bolts.models.autoencoders import seq_AE
from module.seq_datamodule import MRIDataModule



def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size_", default=128, type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset", type=str, default="/data/cifar10")
    parser.add_argument("--subsamp", type=int, default=5)
    parser.add_argument("--seq_jump", type=int, default=5)
    parser.add_argument("--mixed_training", type=int, default=0)
    # parser.add_argument("--data_dir", type=str, default="/data/cifar10")
    # parser.add_argument("--num_workers", type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = seq_AE.add_model_specific_args(parser)
    # parser = AE.add_model_specific_args(parser)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # data, model
    # ------------
    datamodule = None
    if args.dataset == "sequence":
        data_module = MRIDataModule(
            batch_size=args.batch_size, num_workers=args.num_workers, type="seq", subsamp=args.subsamp, seq_jump=args.seq_jump
        )
    elif args.dataset == "image":
        ty = "img_cnn" if "cnn" in args.enc_type else "img_seq"
        data_module = MRIDataModule(
            batch_size=args.batch_size, num_workers=args.num_workers, type=ty, subsamp=args.subsamp, seq_jump=args.seq_jump
        )
    else:
        raise NameError("wrong dataset name")

    if "fc" in args.enc_type:
        enc_out_dim = 300 # 512,
        latent_dim = 2  # T1 and T2,
        input_size = (200,) #(1000//args.subsamp,)  # self.L//subsamp of bloch decoder, (200,)
    elif "cnn" in args.enc_type:
        enc_out_dim = None
        latent_dim = (3, 128, 128)
        input_size = (50, 128, 128)
    else:
        raise NameError("wrong model name")

    model = seq_AE(
        enc_type=args.enc_type,
        enc_out_dim=enc_out_dim,
        latent_dim=latent_dim,
        input_size=input_size,
        emb_lambda=args.emb_lambda,
        max_epochs=args.max_epochs,
        data_type=args.dataset,
        subsamp=args.subsamp,
        seq_jump=args.seq_jump,
        mixed_training=args.mixed_training,
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
            mode="debug"
            # "sync"
        )  # mode="debug",
        neptune_logger = NeptuneLogger(run=run)
        dirpath = os.path.join(args.default_root_dir, neptune_logger.version)

        # ------------
        # callbacks
        # ------------
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val_loss",
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
