try:

    import torch
    import torch.nn as nn
    import torch.optim
    import torch.utils.data
    from torch.utils.data import DataLoader
    import pandas as pd
    from pathlib import Path
    import pytorch_lightning as pl
    import numpy as np
    import cv2
    from lib.lib_utils import Utils

except Exception as e:

    print("Some module are missing {}".format(e))


class MyRegressor(pl.LightningModule):
    def __init__(self, cfg, config=None):
        super(MyRegressor, self).__init__()

        self.learning_rate = cfg.train.base_lr if config is None else config["lr"]
        self.normalize = cfg.normalize
        self.step_size = cfg.train.step_size_lr
        self.count = 0
        self.errors = []
        self.plot_y = []
        self.plot_y_hat = []

        if self.normalize == "z_score":
            mean_std = np.loadtxt(
                str(
                    Path(cfg.train.spath).joinpath("train", "mean_std_total_energy.txt")
                )
            )
            self.mean, self.std = mean_std[0], mean_std[1]
        if self.normalize == "normalization":
            min_max = np.loadtxt(
                str(Path(cfg.train.spath).joinpath("train", "min_max_total_energy.txt"))
            )
            self.min, self.max = min_max[0], min_max[1]

        self.min_val_loss = float("inf")

        # self.net = MySimpleNet(
        #     resolution=cfg.resolution,
        #     input_channels=cfg.atom_types,
        #     output_channels=(cfg.atom_types + 1),
        # )
        self.net = MySimpleResNet(
            resolution=cfg.resolution,
            input_channels=cfg.atom_types,
            output_channels=(cfg.atom_types + 1),
        )
        # self.net = DeepCNN(
        #     resolution=cfg.resolution,
        #     input_channels=cfg.atom_types,
        #     output_channels=(cfg.atom_types + 1),
        # )

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
                    opt, patience=25
                ),
                "monitor": "val_loss",
            },
        }

    def criterion(self, output, target, data):
        l2 = nn.MSELoss()
        tot = output[:, 0] + data[:] * output[:, 1]

        if self.normalize == "z_score":
            target = (target - self.mean) / self.std
        if self.normalize == "normalization":
            target = (target - self.min) / (self.max - self.min)

        return torch.sqrt(l2(tot, target))

    def accuracy(self, output, target, data, test_step=False):
        tot = output[:, 0] + data[:] * output[:, 1]

        if self.normalize == "z_score":
            tot = (tot * self.std) + self.mean
        if self.normalize == "normalization":
            tot = tot * (self.max - self.min) + self.min

        error = torch.abs(tot - target) / torch.abs(target) * 100.0

        if test_step:
            return error, tot
        else:
            return torch.mean(100.0 - error)

    def training_step(self, train_batch, batch_idx=None, optimizer_idx=None):
        x, n_atoms, y = train_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )
        self.log(
            "train_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )

        return loss

    def validation_step(self, val_batch, batch_idx=None):
        x, n_atoms, y = val_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y, n_atoms)
        acc = self.accuracy(y_hat, y, n_atoms)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )
        self.log(
            "val_acc", acc, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )

        return loss

    def test_step(self, test_batch, batch_idx=None):
        x, n_atoms, y = test_batch
        y_hat = self(x)
        error, predictions = self.accuracy(y_hat, y, n_atoms, test_step=True)

        self.errors = [*self.errors, *error.tolist()]
        self.plot_y = [*self.plot_y, *y.tolist()]
        self.plot_y_hat = [*self.plot_y_hat, *predictions.tolist()]

    def validation_epoch_end(self, outputs):
        loss = sum(output for output in outputs) / len(outputs)
        self.count += 1
        if self.min_val_loss > loss:
            print(
                f"In epoch {self.current_epoch} reached a new minimum for validation loss: {loss}, patience: {self.count} epochs"
            )
            self.min_val_loss = loss
            self.count = 0

    def on_test_start(self):
        self.errors.clear()
        self.plot_y.clear()
        self.plot_y_hat.clear()


class MyDataloader(pl.LightningDataModule):
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

    def setup(self, stage=None):
        print(stage)

        train_dataset = pd.read_csv(Path(self.spath).joinpath("train", "train.csv"))
        val_dataset = pd.read_csv(Path(self.spath).joinpath("val", "val.csv"))
        test_dataset = pd.read_csv(Path(self.spath).joinpath("test", "test.csv"))

        # collect the path of the .npy files for each set in order to generate the DataLoader objects
        train_paths = []
        val_paths = []
        test_paths = []

        for i in range(len(train_dataset["file_name"])):
            train_paths.append(
                Path(self.spath).joinpath(
                    "train", train_dataset["file_name"][i] + ".png"
                )
            )

        for i in range(len(val_dataset["file_name"])):
            val_paths.append(
                Path(self.spath).joinpath("val", val_dataset["file_name"][i] + ".png")
            )

        for i in range(len(test_dataset["file_name"])):
            test_paths.append(
                Path(self.spath).joinpath("test", test_dataset["file_name"][i] + ".png")
            )

        self.train_data = MyDatasetPng(
            train_paths,
            train_dataset[self.target],
            resolution=self.resolution,
        )
        self.val_data = MyDatasetPng(
            val_paths,
            val_dataset[self.target],
            resolution=self.resolution,
        )
        self.test_data = MyDatasetPng(
            test_paths,
            test_dataset[self.target],
            resolution=self.resolution,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            if not self.cluster
            else self.cluster_num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
            if not self.cluster
            else self.cluster_num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
            if not self.cluster
            else self.cluster_num_workers,
            pin_memory=True,
            drop_last=True,
        )


class MySimpleNet(nn.Module):
    def __init__(self, resolution=160, input_channels=1, output_channels=2):
        super(MySimpleNet, self).__init__()

        self.conv128 = nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=3
        )
        self.batchnorm1 = nn.BatchNorm2d(self.conv128.out_channels)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv256_1 = nn.Conv2d(
            in_channels=self.conv128.out_channels, out_channels=128, kernel_size=3
        )
        self.conv256_2 = nn.Conv2d(
            in_channels=self.conv256_1.out_channels, out_channels=128, kernel_size=3
        )
        self.batchnorm2 = nn.BatchNorm2d(self.conv256_2.out_channels)
        self.conv512_1 = nn.Conv2d(
            in_channels=self.conv256_2.out_channels, out_channels=256, kernel_size=3
        )
        self.conv512_2 = nn.Conv2d(
            in_channels=self.conv512_1.out_channels, out_channels=256, kernel_size=3
        )
        self.batchnorm3 = nn.BatchNorm2d(self.conv512_2.out_channels)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.find_dimenstion(resolution, input_channels), 512)
        self.batchnorm4 = nn.BatchNorm1d(self.fc1.out_features)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.batchnorm5 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, output_channels)

    def forward(self, input):

        x = self.relu(self.conv128(input))
        x = self.batchnorm1(x)
        x = self.max_pool(x)
        x = self.relu(self.conv256_1(x))
        x = self.relu(self.conv256_2(x))
        x = self.batchnorm2(x)
        x = self.max_pool(x)
        x = self.relu(self.conv512_1(x))
        x = self.relu(self.conv512_2(x))
        x = self.batchnorm3(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm5(x)

        return self.fc3(x)

    def find_dimenstion(self, resolution, input_channels):

        x = torch.rand(1, input_channels, resolution, resolution)

        x = self.relu(self.conv128(x))
        x = self.batchnorm1(x)
        x = self.max_pool(x)
        x = self.relu(self.conv256_1(x))
        x = self.relu(self.conv256_2(x))
        x = self.batchnorm2(x)
        x = self.max_pool(x)
        x = self.relu(self.conv512_1(x))
        x = self.relu(self.conv512_2(x))
        x = self.batchnorm3(x)
        x = self.max_pool(x)
        x = self.flatten(x)

        return x.size()[1]


class MySimpleResNet(nn.Module):
    def __init__(self, resolution=160, input_channels=1, output_channels=2):
        super(MySimpleResNet, self).__init__()

        self.conv128 = nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=3
        )
        self.batchnorm1 = nn.BatchNorm2d(self.conv128.out_channels)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv256_1 = nn.Conv2d(
            in_channels=self.conv128.out_channels,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv256_2 = nn.Conv2d(
            in_channels=self.conv256_1.out_channels,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm2 = nn.BatchNorm2d(self.conv256_2.out_channels)
        self.conv512_1 = nn.Conv2d(
            in_channels=self.conv256_2.out_channels,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.conv512_2 = nn.Conv2d(
            in_channels=self.conv512_1.out_channels,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm3 = nn.BatchNorm2d(self.conv512_2.out_channels)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.fc1 = nn.Linear(self.find_dimenstion(resolution, input_channels), 512)
        self.batchnorm4 = nn.BatchNorm1d(self.fc1.out_features)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.batchnorm5 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, output_channels)

    def forward(self, input):

        x = self.conv128(input)
        x = self.relu(self.batchnorm1(x))
        x = self.max_pool(x)
        residual = x
        x = self.conv256_1(x)
        x = self.conv256_2(x)
        x = self.relu(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool(x)
        residual = x
        x = self.conv512_1(x)
        x = self.conv512_2(x)
        x = self.relu(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(self.batchnorm4(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(self.batchnorm5(x))

        return self.fc3(x)

    def find_dimenstion(self, resolution, input_channels):

        x = torch.rand(1, input_channels, resolution, resolution)

        x = self.conv128(x)
        x = self.relu(self.batchnorm1(x))
        x = self.max_pool(x)
        residual = x
        x = self.conv256_1(x)
        x = self.conv256_2(x)
        x = self.relu(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool(x)
        residual = x
        x = self.conv512_1(x)
        x = self.conv512_2(x)
        x = self.relu(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool(x)
        x = self.flatten(x)

        return x.size()[1]


class DeepCNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(
        self,
        resolution=160,
        input_channels=1,
        output_channels=1,
    ):
        super(DeepCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(
            self.find_dimenstion(resolution, input_channels), 128, bias=False
        )
        self.relu1 = nn.ReLU()
        self.dropout2d = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, output_channels, bias=False)

    # Progresses data across layers
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.max_pool1(out)
        # out = self.dropout2d(out)

        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.max_pool2(out)
        # out = self.dropout2d(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.max_pool3(out)
        # out = self.dropout2d(out)

        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.max_pool4(out)
        # out = self.dropout2d(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        # out = self.dropout(out)
        out = self.fc2(out)

        return out

    def find_dimenstion(self, resolution, input_channels):

        x = torch.rand(1, input_channels, resolution, resolution)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.max_pool1(out)
        # out = self.dropout2d(out)

        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.max_pool2(out)
        # out = self.dropout2d(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.max_pool3(out)
        # out = self.dropout2d(out)

        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.max_pool4(out)
        # out = self.dropout2d(out)

        out = out.reshape(out.size(0), -1)

        return out.size()[1]


class MyDatasetPng:
    """Class that generate a dataset for DataLoader module, given as input the paths of the .png files and the respective labels"""

    def __init__(
        self,
        paths,
        targets,
        padding=True,
        resolution=160,
    ):
        self.paths = paths
        self.targets = targets
        self.padding = padding
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), 0)
        if self.padding:
            img = Utils.padding_image(img, size=self.resolution)
        img = np.asarray(img, float) / 255.0

        target = np.array(float(self.targets[i]))

        n_atoms = np.loadtxt((self.paths[i]).with_suffix(".txt"))

        return (
            torch.from_numpy(np.expand_dims(img.copy(), 0)).float(),
            torch.from_numpy(n_atoms).float(),
            torch.from_numpy(target).float(),
        )


class MyDatasetPngCluster:
    """Class that generate a dataset for DataLoader module, given as input the paths of the .png files and the respective labels"""

    def __init__(
        self,
        paths,
        targets,
        padding=True,
        resolution=160,
    ):
        self.paths = paths
        self.targets = targets
        self.padding = padding
        self.resolution = resolution

        self.images = []
        self.properties = []
        self.n_atoms = []

        for path, target in self.paths, self.targets:
            img = cv2.imread(str(path), 0)
            if self.padding:
                img = Utils.padding_image(img, size=self.resolution)
            img = np.asarray(img, float) / 255.0

            target = np.array(float(target))

            n_atoms = np.loadtxt(path.with_suffix(".txt"))

            self.images.append(torch.from_numpy(np.expand_dims(img.copy(), 0)).float())
            self.properties.append(torch.from_numpy(n_atoms).float())
            self.n_atoms.append(torch.from_numpy(target).float())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return (
            self.images[i],
            self.properties[i],
            self.n_atoms[i],
        )


if __name__ == "__main__":
    pass
