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
    from lib.lib_networks import (
        InceptionResNet,
        MySimpleNet,
        MySimpleResNet,
        DeepCNN,
        MyDatasetPng,
        MyDatasetCoulomb,
        get_resnet_model,
    )

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class MyRegressor(LightningModule):
    def __init__(self, cfg, config=None):
        super(MyRegressor, self).__init__()

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

        self.net = InceptionResNet(
            resolution=cfg.resolution,
            input_channels=3 if (not self.coulomb and self.atom_types > 1) else 1,
            output_channels=(
                (self.atom_types + 1)
                if (self.target == "total_energy" or self.target == "formation_energy")
                else 1
            ),
            filters=[16, 32, 64],
            dense_layers=[128, 64],
        )

        self.save_hyperparameters()

    def forward(self, x):
        out = self.net(x)

        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

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

    def criterion(self, output, target, data):
        l2 = nn.MSELoss()
        if self.target == "total_energy" or self.target == "formation_energy":
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
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.train_loss_step_holder.append(loss)
        self.train_acc_step_holder.append(acc)

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        x, n_atoms, y = val_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.val_loss_step_holder.append(loss)
        self.val_acc_step_holder.append(acc)

        return loss

    def test_step(self, test_batch, batch_idx=None):
        x, n_atoms, y, names = test_batch
        y_hat = self(x)
        error, predictions = self.accuracy(y_hat, y, n_atoms, test_step=True)

        self.errors = [*self.errors, *error.tolist()]
        self.plot_y = [*self.plot_y, *y.tolist()]
        self.plot_y_hat = [*self.plot_y_hat, *predictions.tolist()]
        self.sample_names = [*self.sample_names, *names]

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_loss_step_holder).mean(dim=0)
        acc = torch.stack(self.val_acc_step_holder).mean(dim=0)

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
        self.package_path = Path(__file__).parent.parent
        self.spath = self.package_path.joinpath(cfg.train.dataset_path)
        self.target = cfg.target

        self.batch_size = (
            cfg.train.batch_size if config is None else config["batch_size"]
        )

        self.resolution = cfg.resolution
        self.num_workers = cfg.num_workers
        self.enlargement_method = cfg.enlargement_method
        self.coulomb = cfg.coulomb

    def setup(self, stage=None):
        train_dataset = pd.read_csv(Path(self.spath).joinpath("train", "train.csv"))
        val_dataset = pd.read_csv(Path(self.spath).joinpath("val", "val.csv"))
        test_dataset = pd.read_csv(Path(self.spath).joinpath("test", "test.csv"))

        # collect the path of the .npy files for each set in order to generate the DataLoader objects
        train_paths = [
            f
            for f in Path(self.spath).joinpath("train").iterdir()
            if (f.suffix == ".png" or f.suffix == ".npy")
        ]
        val_paths = [
            f
            for f in Path(self.spath).joinpath("val").iterdir()
            if (f.suffix == ".png" or f.suffix == ".npy")
        ]
        test_paths = [
            f
            for f in Path(self.spath).joinpath("test").iterdir()
            if (f.suffix == ".png" or f.suffix == ".npy")
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
            )
            self.val_data = MyDatasetPng(
                val_paths,
                val_dataset,
                self.target,
                resolution=self.resolution,
                enlargement_method=self.enlargement_method,
                phase="val",
            )
            self.test_data = MyDatasetPng(
                test_paths,
                test_dataset,
                self.target,
                resolution=self.resolution,
                enlargement_method=self.enlargement_method,
                phase="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=(self.num_workers),
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=(self.num_workers),
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=(self.num_workers),
            pin_memory=True,
            drop_last=True,
        )
