try:
    import torch
    import torch.nn as nn
    import torch.optim
    import torch.utils.data
    from torch.utils.data import DataLoader
    import pandas as pd
    from pathlib import Path
    from lightning import LightningModule, LightningDataModule
    from lightning.pytorch.callbacks import RichProgressBar
    from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
    import numpy as np
    from lib.lib_networks import (
        InceptionResNet,
        MySimpleNet,
        MyDatasetPng,
        MyDatasetCoulomb,
        get_resnet_model,
        CoulombNet,
    )
    from icecream import ic
    from matgl.models import M3GNet
    from lib.lib_equivariant_networks import EquivariantInceptionResNet
    from e2cnn import gspaces
    from e2cnn import nn as e2nn

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class MyRegressor(LightningModule):
    def __init__(self, cfg, config=None):
        super(MyRegressor, self).__init__()

        self.cfg = cfg

        self.learning_rate = cfg.train.base_lr if config is None else config["lr"]
        self.target = cfg.target
        self.atom_types = cfg.atom_types
        self.count = 0
        self.errors = []
        self.plot_y = []
        self.plot_y_hat = []
        self.sample_names = []
        self.coulomb = cfg.coulomb
        self.num_epochs = cfg.train.num_epochs
        self.batch_size = cfg.train.batch_size

        self.val_loss_step_holder = []
        self.val_acc_step_holder = []
        self.train_loss_step_holder = []
        self.train_acc_step_holder = []
        self.compiled = cfg.train.compile

        self.min_val_loss = float("inf")

        self.grayscale = cfg.train.grayscale

        if self.cfg.train.network == "CNN":
            self.net = MySimpleNet(
                resolution=cfg.resolution,
                input_channels=3 if (not self.coulomb and self.atom_types > 1) else 1,
                output_channels=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
            )

        elif self.cfg.train.network == "InceptionResNet":
            self.net = InceptionResNet(
                resolution=cfg.resolution,
                input_channels=(
                    3
                    if (not (self.coulomb or self.grayscale) and self.atom_types > 1)
                    else 1
                ),
                output_channels=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
                filters=[16, 32, 64],
                dense_layers=[128, 64],
            )

        elif self.cfg.train.network == "Resnet18":
            self.net = get_resnet_model(
                in_channels=3 if (not self.coulomb and self.atom_types > 1) else 1,
                out_channels=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
            )

        elif self.cfg.train.network == "CoulombNet":
            self.net = CoulombNet(
                resolution=cfg.resolution,
                output_channels=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
            )

        elif self.cfg.train.network == "M3GNet":
            self.net = M3GNet(
                cutoff=4.0,
                element_types=["H", "C", "O"],
                ntargets=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
            )

        elif self.cfg.train.network == "E2InceptionResNet":
            self.r2_act = gspaces.FlipRot2dOnR2(N=4)
            self.net = EquivariantInceptionResNet(
                resolution=cfg.resolution,
                input_channels=(
                    3
                    if (not (self.coulomb or self.grayscale) and self.atom_types > 1)
                    else 1
                ),
                output_channels=(
                    (self.atom_types + 1)
                    if (
                        self.target == "total_energy"
                        or self.target == "formation_energy"
                    )
                    else 1
                ),
                filters=[16, 32, 64],
                dense_layers=[128, 64],
                r2_act=self.r2_act,
            )

        else:
            raise Exception(f"Network {self.cfg.train.network} not found!")

        self.train_loss_plot = []
        self.train_acc_plot = []
        self.val_loss_plot = []
        self.val_acc_plot = []

        self.save_hyperparameters()

    def forward(self, x):
        out = self.net(x)

        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        if not self.cfg.kfold:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        patience=20,
                        verbose=True,
                    ),
                    "monitor": "val_loss",
                },
            }
        else:
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=33),
                },
            }

    def criterion(self, output, target, data):
        l2 = nn.MSELoss()
        if self.target == "total_energy" or self.target == "formation_energy":
            output = torch.squeeze(output)
            if self.atom_types == 1:
                output = output[:, 0] + data[:] * output[:, 1]
            elif self.atom_types == 2:
                output = (
                    output[:, 0] + data[:, 0] * output[:, 1] + data[:, 1] * output[:, 2]
                )
            elif self.atom_types == 3:
                output = (
                    output[:, 0]
                    + data[:, 0] * output[:, 1]
                    + data[:, 1] * output[:, 2]
                    + data[:, 2] * output[:, 3]
                )
            else:
                raise Exception("Wrong number of atom types\n")
        else:
            output = torch.squeeze(output)

        return (
            torch.sqrt(l2(output, target))
            if (self.target == "total_energy" or self.target == "formation_energy")
            else l2(output, target)
        )

    def accuracy(self, output, target, data, test_step=False):
        if self.target == "total_energy" or self.target == "formation_energy":
            output = torch.squeeze(output)
            if self.atom_types == 1:
                output = output[:, 0] + data[:] * output[:, 1]
            elif self.atom_types == 2:
                output = (
                    output[:, 0] + data[:, 0] * output[:, 1] + data[:, 1] * output[:, 2]
                )
            elif self.atom_types == 3:
                output = (
                    output[:, 0]
                    + data[:, 0] * output[:, 1]
                    + data[:, 1] * output[:, 2]
                    + data[:, 2] * output[:, 3]
                )
            else:
                raise Exception("Wrong number of atom types\n")
        else:
            output = torch.squeeze(output)

        error = torch.abs(output - target) / torch.abs(target) * 100.0

        if test_step:
            return error, output
        else:
            return torch.mean(100.0 - error)

    def training_step(self, train_batch, batch_idx=None):
        x, n_atoms, y = train_batch
        if self.cfg.train.network == "E2InceptionResNet":
            x = e2nn.GeometricTensor(
                x, e2nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
            )
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.train_loss_step_holder.append(loss)
        self.train_acc_step_holder.append(acc)

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        x, n_atoms, y = val_batch
        if self.cfg.train.network == "E2InceptionResNet":
            x = e2nn.GeometricTensor(
                x, e2nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
            )
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.val_loss_step_holder.append(loss)
        self.val_acc_step_holder.append(acc)

        return loss

    def test_step(self, test_batch, batch_idx=None):
        x, n_atoms, y, names = test_batch
        if self.cfg.train.network == "E2InceptionResNet":
            x = e2nn.GeometricTensor(
                x, e2nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
            )
        y_hat = self(x)
        error, predictions = self.accuracy(y_hat, y, n_atoms, test_step=True)

        self.errors = [*self.errors, *error.tolist()]
        self.plot_y = [*self.plot_y, *y.tolist()]
        self.plot_y_hat = [*self.plot_y_hat, *predictions.tolist()]
        self.sample_names = [*self.sample_names, *names]

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.val_acc_step_holder).mean(dim=0)

        self.val_loss_plot.append(loss.item())
        self.val_acc_plot.append(acc.item())

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )
        if self.current_epoch > 0:
            loss_difference = (
                abs(self.train_loss_plot[-1] - self.val_loss_plot[-1])
                + self.val_loss_plot[-1]
            )
        else:
            loss_difference = np.inf
        self.log(
            "loss_difference",
            loss_difference,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )

        self.count += 1
        if self.min_val_loss > loss:
            print(
                f"In epoch {self.current_epoch} reached a new minimum for validation loss: {loss}, patience: {self.count} epochs"
            )
            self.min_val_loss = loss
            self.count = 0

        self.val_loss_step_holder.clear()
        self.val_acc_step_holder.clear()

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.train_acc_step_holder).mean(dim=0)

        self.train_loss_plot.append(loss.item())
        self.train_acc_plot.append(acc.item())

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
            sync_dist=True,
        )

        self.train_loss_step_holder.clear()
        self.train_acc_step_holder.clear()

        ic(f"Ended epoch: {self.current_epoch}")

    def on_test_start(self):
        self.errors.clear()
        self.plot_y.clear()
        self.plot_y_hat.clear()
        self.sample_names.clear()

    def on_train_start(self):
        self.log_dict(
            {
                "hp/num_epochs": float(self.num_epochs),
                "hp/learning_rate": float(self.learning_rate),
                "hp/batch_size": float(self.batch_size),
            }
        )
        self.train_loss_plot.clear()
        self.train_acc_plot.clear()
        self.val_loss_plot.clear()
        self.val_acc_plot.clear()

    @staticmethod
    def get_progressbar():
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="#e809a1",
                progress_bar="#6206E0",
                progress_bar_finished="#00c900",
                progress_bar_pulse="#6206E0",
                batch_progress="#e809a1",
                time="#e8c309",
                processing_speed="#e8c309",
                metrics="#dbd7d7",
            ),
        )

        return progress_bar


class MyDataloader(LightningDataModule):
    def __init__(self, cfg, config=None):
        super().__init__()
        self.spath = cfg.train.spath
        self.target = cfg.target

        self.batch_size = (
            cfg.train.batch_size if config is None else config["batch_size"]
        )

        self.resolution = cfg.resolution
        self.num_workers = cfg.num_workers
        self.cluster = cfg.cluster
        self.cluster_num_workers = cfg.cluster_num_workers
        self.enlargement_method = cfg.enlargement_method
        self.coulomb = cfg.coulomb

        self.cfg = cfg

    def setup(self, stage=None):
        train_dataset = pd.read_csv(Path(self.spath).joinpath("train", "train.csv"))
        val_dataset = pd.read_csv(Path(self.spath).joinpath("val", "val.csv"))
        test_dataset = pd.read_csv(Path(self.spath).joinpath("test", "test.csv"))

        # train_dataset[self.target] = (
        #     train_dataset[self.target] - train_dataset[self.target].min()
        # ) / (train_dataset[self.target].max() - train_dataset[self.target].min())
        # val_dataset[self.target] = (
        #     val_dataset[self.target] - val_dataset[self.target].min()
        # ) / (val_dataset[self.target].max() - val_dataset[self.target].min())
        # test_dataset[self.target] = (
        #     test_dataset[self.target] - test_dataset[self.target].min()
        # ) / (test_dataset[self.target].max() - test_dataset[self.target].min())

        # collect the path of the .npy files for each set in order to generate the DataLoader objects
        train_paths = [
            f
            for f in Path(self.spath).joinpath("train").iterdir()
            if (
                (f.suffix == ".png" and not self.cfg.coulomb)
                or (f.suffix == ".npy" and self.cfg.coulomb)
            )
        ]
        val_paths = [
            f
            for f in Path(self.spath).joinpath("val").iterdir()
            if (
                (f.suffix == ".png" and not self.cfg.coulomb)
                or (f.suffix == ".npy" and self.cfg.coulomb)
            )
        ]
        test_paths = [
            f
            for f in Path(self.spath).joinpath("test").iterdir()
            if (
                (f.suffix == ".png" and not self.cfg.coulomb)
                or (f.suffix == ".npy" and self.cfg.coulomb)
            )
        ]

        if self.coulomb:
            self.train_data = MyDatasetCoulomb(
                train_paths,
                train_dataset,
                self.target,
                resolution=self.resolution,
                phase="train",
            )
            self.val_data = MyDatasetCoulomb(
                val_paths,
                val_dataset,
                self.target,
                resolution=self.resolution,
                phase="val",
            )
            self.test_data = MyDatasetCoulomb(
                test_paths,
                test_dataset,
                self.target,
                resolution=self.resolution,
                phase="test",
            )
        else:
            self.train_data = MyDatasetPng(
                train_paths,
                train_dataset,
                self.target,
                resolution=self.resolution,
                enlargement_method=self.enlargement_method,
                phase="train",
                grayscale=self.cfg.train.grayscale,
            )
            self.val_data = MyDatasetPng(
                val_paths,
                val_dataset,
                self.target,
                resolution=self.resolution,
                enlargement_method=self.enlargement_method,
                phase="val",
                grayscale=self.cfg.train.grayscale,
            )
            self.test_data = MyDatasetPng(
                test_paths,
                test_dataset,
                self.target,
                resolution=self.resolution,
                enlargement_method=self.enlargement_method,
                phase="test",
                grayscale=self.cfg.train.grayscale,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=(
                self.num_workers if not self.cluster else self.cluster_num_workers
            ),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=(
                self.num_workers if not self.cluster else self.cluster_num_workers
            ),
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=(
                self.num_workers if not self.cluster else self.cluster_num_workers
            ),
            pin_memory=True,
            drop_last=True,
        )
