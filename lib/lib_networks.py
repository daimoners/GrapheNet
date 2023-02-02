try:

    import torch.nn as nn
    import torch
    import cv2
    import numpy as np
    from lib.lib_utils import Utils

except Exception as e:

    print(f"Some module are missing: {e}")


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # Apply the convolutional layers and max pooling in parallel
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        # Concatenate the resulting feature maps along the channel axis
        x_out = torch.cat([x1, x3, x5], dim=1)
        return x_out


class InceptionResNet(nn.Module):
    def __init__(
        self,
        resolution: int = 160,
        input_channels: int = 3,
        output_channels: int = 2,
        filters: list = [64, 128, 256],
        dense_layers: list = [512, 256],
    ):
        super(InceptionResNet, self).__init__()

        self.inception1 = InceptionBlock(
            in_channels=input_channels, out_channels=filters[0]
        )
        self.inception2 = InceptionBlock(
            in_channels=(filters[0] * 3), out_channels=filters[1]
        )
        self.inception3 = InceptionBlock(
            in_channels=(filters[1] * 3), out_channels=filters[2]
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d((filters[0] * 3))
        self.batchnorm2 = nn.BatchNorm2d((filters[1] * 3))
        self.batchnorm3 = nn.BatchNorm2d((filters[2] * 3))

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(
                (filters[0] + filters[1]),
                2 * (filters[0] + filters[1]),
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * (filters[0] + filters[1])),
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(
                (filters[1] + filters[2]),
                2 * (filters[1] + filters[2]),
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(2 * (filters[1] + filters[2])),
        )

        self.fc1 = nn.Linear(
            self.find_dimenstion(resolution, input_channels), dense_layers[0]
        )
        self.batchnorm4 = nn.BatchNorm1d(dense_layers[0])
        self.fc2 = nn.Linear(dense_layers[0], dense_layers[1])
        self.batchnorm5 = nn.BatchNorm1d(dense_layers[1])
        self.fc3 = nn.Linear(dense_layers[1], output_channels)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.inception1(x)
        x = self.relu(self.batchnorm1(x))
        x = self.max_pool(x)

        residual = x

        x = self.inception2(x)
        x = self.relu(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool(x)

        residual = x

        x = self.inception3(x)
        x = self.relu(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(self.batchnorm4(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(self.batchnorm5(x))
        x = self.fc3(x)

        return x

    def find_dimenstion(self, resolution, input_channels):

        x = torch.rand(1, input_channels, resolution, resolution)

        x = self.inception1(x)
        x = self.relu(self.batchnorm1(x))
        x = self.max_pool(x)

        residual = x

        x = self.inception2(x)
        x = self.relu(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool(x)

        residual = x
        x = self.inception3(x)
        x = self.relu(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool(x)

        x = self.flatten(x)

        return x.size()[1]


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


class MySimpleResNetWithDistribution(nn.Module):
    def __init__(self, resolution=160, input_channels=1, output_channels=2):
        super(MySimpleResNetWithDistribution, self).__init__()

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

    def forward(self, input1, input2):

        x = self.conv128(input1)
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
        x = self.fc1(
            torch.cat((x, input2), dim=1)
        )  # TODO forse va concatenato quando la dimensione della prima parte è 3/4 volte la dimensione della distribuzione (24/32 se la dimensione della distribuzione è 8)
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

        return x.size()[1] + 8


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
        resolution=160,
        enlargement_method="padding",
    ):
        self.paths = paths
        self.targets = targets
        self.resolution = resolution
        self.enlargement_method = enlargement_method

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), -1)
        if self.enlargement_method == "padding":
            img = Utils.padding_image(img, size=self.resolution)
        elif self.enlargement_method == "resize":
            img = cv2.resize(
                img, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC
            )
        img = np.asarray(img, float) / 255.0

        target = np.array(float(self.targets[i]))

        n_atoms = np.loadtxt((self.paths[i]).with_suffix(".txt"))

        if len(img.shape) == 2:
            return (
                torch.from_numpy(np.expand_dims(img.copy(), 0)).float(),
                torch.from_numpy(n_atoms).float(),
                torch.from_numpy(target).float(),
            )
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return (
                torch.squeeze(
                    torch.from_numpy(np.expand_dims(img.copy(), 0))
                    .permute(0, 3, 1, 2)
                    .float()
                ),
                torch.from_numpy(n_atoms).float(),
                torch.from_numpy(target).float(),
            )
        else:
            raise Exception("Wrong dimensions for the input images\n")


class MyDatasetPngCluster:
    """Class that generate a dataset for DataLoader module, given as input the paths of the .png files and the respective labels"""

    def __init__(
        self,
        paths,
        targets,
        resolution=160,
        enlargement_method="padding",
    ):
        self.paths = paths
        self.targets = targets
        self.resolution = resolution
        self.enlargement_method = enlargement_method

        self.images = []
        self.properties = []
        self.n_atoms = []

        for path, target in zip(self.paths, self.targets):
            img = cv2.imread(str(path), -1)
            if self.enlargement_method == "padding":
                img = Utils.padding_image(img, size=self.resolution)
            elif self.enlargement_method == "resize":
                img = cv2.resize(
                    img,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_CUBIC,
                )
            img = np.asarray(img, float) / 255.0

            target = np.array(float(target))

            n_atoms = np.loadtxt(path.with_suffix(".txt"))

            if len(img.shape) == 2:
                self.images.append(
                    torch.from_numpy(np.expand_dims(img.copy(), 0)).float()
                )
                self.properties.append(torch.from_numpy(n_atoms).float())
                self.n_atoms.append(torch.from_numpy(target).float())
            elif len(img.shape) == 3 and img.shape[2] == 3:
                self.images.append(
                    torch.squeeze(
                        torch.from_numpy(np.expand_dims(img.copy(), 0))
                        .permute(0, 3, 1, 2)
                        .float()
                    )
                )
                self.properties.append(torch.from_numpy(n_atoms).float())
                self.n_atoms.append(torch.from_numpy(target).float())
            else:
                raise Exception("Wrong dimensions for the input images\n")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return (
            self.images[i],
            self.properties[i],
            self.n_atoms[i],
        )


class MyDatasetPngWithDistribution:
    """Class that generate a dataset for DataLoader module, given as input the paths of the .png files and the respective labels"""

    def __init__(
        self,
        paths,
        targets,
        resolution=160,
        enlargement_method="padding",
    ):
        self.paths = paths
        self.targets = targets
        self.resolution = resolution
        self.enlargement_method = enlargement_method

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), -1)
        if self.enlargement_method == "padding":
            img = Utils.padding_image(img, size=self.resolution)
        elif self.enlargement_method == "resize":
            img = cv2.resize(
                img, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC
            )
        img = np.asarray(img, float) / 255.0

        target = np.array(float(self.targets[i]))

        n_atoms = np.loadtxt((self.paths[i]).with_suffix(".txt"))

        distribution = np.loadtxt(
            (self.paths[i]).with_name(f"{(self.paths[i]).stem}_distribution.txt")
        )
        norm = np.linalg.norm(distribution)
        distribution_normalized = distribution / norm

        if len(img.shape) == 2:
            return (
                torch.from_numpy(np.expand_dims(img.copy(), 0)).float(),
                torch.from_numpy(n_atoms).float(),
                torch.from_numpy(target).float(),
                torch.from_numpy(distribution_normalized).float(),
            )
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return (
                torch.squeeze(
                    torch.from_numpy(np.expand_dims(img.copy(), 0))
                    .permute(0, 3, 1, 2)
                    .float()
                ),
                torch.from_numpy(n_atoms).float(),
                torch.from_numpy(target).float(),
                torch.from_numpy(distribution_normalized).float(),
            )
        else:
            raise Exception("Wrong dimensions for the input images\n")


if __name__ == "__main__":
    x = torch.rand(32, 3, 160, 160)

    net = InceptionResNet(input_channels=3, output_channels=4)

    y = net(x)

    print(y.shape)
