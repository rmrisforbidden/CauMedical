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


# Example
"""
# Basic run
from pl_bolts.models.autoencoders import AE
model = AE()
trainer = Trainer()
trainer.fit(model)

# Change Encoder
from pl_bolts.models.autoencoders import AE
class MyAEFlavor(AE):
    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
        encoder = YourSuperFancyEncoder(...)
        return encoder

# Pretrained Encoder
from pl_bolts.models.autoencoders import AE
ae = AE(input_height=32)
print(AE.pretrained_weights_available())
ae = ae.from_pretrained('cifar10-resnet18')
ae.freeze()
"""


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size_", default=128, type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset", type=str, default="/data/cifar10")
    # parser.add_argument("--data_dir", type=str, default="/data/cifar10")
    # parser.add_argument("--num_workers", type=int, default=8)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AE.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # data, model
    # ------------
    datamodule = None
    if args.dataset == "cifar10":
        model = AE(enc_type="resnet18", maxpool1=False, enc_out_dim=512, latent_dim=256, input_height=32)
        args.patch_size = 8
        data_module = CIFAR10DataModule(
            data_dir=args.data_dir, dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.dataset == "brainweb":
        model = AE(enc_type="simple", maxpool1=False, enc_out_dim=254, latent_dim=254, input_height=144)
        args.patch_size = 16
        data_module = BrainWebDataModule(
            data_dir=args.data_dir, dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

    else:
        raise NameError("wrong dataset name")

    # ------------
    # Trainer
    # ------------
    if True:
        API_KEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="
        run = neptune.init(
            api_token=API_KEY, project=f"rmrisforbidden/{args.default_root_dir.split('/')[-1]}", mode="sync"
        )
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

    else:
        # ------------
        # callbacks
        # ------------
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val_loss",
            filename="checkpt-{epoch:02d}-{val_loss:.2f}",
            # filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer.from_argparse_args(
            args,
            progress_bar_refresh_rate=20,
            check_val_every_n_epoch=1,
            precision=16,
            gradient_clip_val=2.0,
            callbacks=[checkpoint_callback, lr_monitor],
        )
        dirpath = args.default_root_dir

    # train
    trainer.fit(model, datamodule=data_module)

    # test
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
