try:
    import torch
    import torch.nn as nn
    from e2cnn import gspaces
    from e2cnn import nn as e2nn

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}")


class EquivariantInceptionBlock(nn.Module):
    def __init__(self, r2_act, in_channels, out_channels, reg=False):
        super(EquivariantInceptionBlock, self).__init__()

        self.r2_act = r2_act

        # Definire il tipo di input e output equivariante
        if reg:
            self.input_type = e2nn.FieldType(
                r2_act, in_channels * [r2_act.regular_repr]
            )
        else:
            self.input_type = e2nn.FieldType(
                r2_act, in_channels * [r2_act.trivial_repr]
            )
        self.output_type = e2nn.FieldType(r2_act, out_channels * [r2_act.regular_repr])

        # Convolutional layers equivarianti
        self.conv1 = e2nn.R2Conv(self.input_type, self.output_type, kernel_size=1)
        self.conv3 = e2nn.R2Conv(
            self.input_type, self.output_type, kernel_size=3, padding=1
        )
        self.conv5 = e2nn.R2Conv(
            self.input_type, self.output_type, kernel_size=5, padding=2
        )

        # ReLU attivazione equivariante per l'output
        self.relu1 = e2nn.ReLU(self.output_type)
        self.relu3 = e2nn.ReLU(self.output_type)
        self.relu5 = e2nn.ReLU(self.output_type)

    def forward(self, x):

        # Applica i layer convoluzionali
        x1 = self.relu1(self.conv1(x))
        x3 = self.relu3(self.conv3(x))
        x5 = self.relu5(self.conv5(x))

        # Concatenare i tensori risultanti
        x_out = e2nn.tensor_directsum(
            [x1, x3, x5]
        )  # Somma diretta per concatenare i tensori equivarianti

        return x_out


class EquivariantInceptionResNet(nn.Module):
    def __init__(
        self,
        resolution: int = 160,
        input_channels: int = 3,
        output_channels: int = 1,
        filters: list = [32, 64, 128],
        dense_layers: list = [256, 128],
        r2_act=None,
    ):
        super(EquivariantInceptionResNet, self).__init__()

        if r2_act is None:
            r2_act = gspaces.FlipRot2dOnR2(N=4)

        self.r2_act = r2_act

        # Definisci i tipi di campo per i layer
        self.input_type = e2nn.FieldType(r2_act, input_channels * [r2_act.trivial_repr])
        self.in_type_1 = e2nn.FieldType(r2_act, filters[0] * 3 * [r2_act.regular_repr])
        self.in_type_2 = e2nn.FieldType(r2_act, filters[1] * 3 * [r2_act.regular_repr])
        self.in_type_3 = e2nn.FieldType(r2_act, filters[2] * 3 * [r2_act.regular_repr])

        self.inception1 = EquivariantInceptionBlock(
            in_channels=input_channels, out_channels=filters[0], r2_act=r2_act
        )
        self.inception2 = EquivariantInceptionBlock(
            in_channels=(filters[0] * 3),
            out_channels=filters[1],
            r2_act=r2_act,
            reg=True,
        )
        self.inception3 = EquivariantInceptionBlock(
            in_channels=(filters[1] * 3),
            out_channels=filters[2],
            r2_act=r2_act,
            reg=True,
        )

        self.relu1 = e2nn.ReLU(self.in_type_1)
        self.relu2 = e2nn.ReLU(self.in_type_2)
        self.relu3 = e2nn.ReLU(self.in_type_3)

        self.batchnorm1 = e2nn.InnerBatchNorm(self.in_type_1)
        self.batchnorm2 = e2nn.InnerBatchNorm(self.in_type_2)
        self.batchnorm3 = e2nn.InnerBatchNorm(self.in_type_3)

        self.max_pool1 = e2nn.PointwiseAvgPool(self.in_type_1, kernel_size=3, padding=1)
        self.max_pool2 = e2nn.PointwiseAvgPool(self.in_type_2, kernel_size=3, padding=1)
        self.max_pool3 = e2nn.PointwiseAvgPool(self.in_type_3, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        # Downsample layers
        self.downsample_0 = nn.Sequential(
            e2nn.R2Conv(
                e2nn.FieldType(r2_act, input_channels * [r2_act.trivial_repr]),
                e2nn.FieldType(r2_act, filters[0] * 3 * [r2_act.regular_repr]),
                kernel_size=1,
                bias=False,
            ),  # nn.Conv2d(input_channels, filters[0] + filters[1], kernel_size=1, bias=False),
            e2nn.InnerBatchNorm(
                e2nn.FieldType(r2_act, filters[0] * 3 * [r2_act.regular_repr])
            ),  # nn.BatchNorm2d(filters[0] + filters[1]),
        )

        self.downsample_1 = nn.Sequential(
            e2nn.R2Conv(
                e2nn.FieldType(r2_act, filters[0] * 3 * [r2_act.regular_repr]),
                e2nn.FieldType(r2_act, filters[1] * 3 * [r2_act.regular_repr]),
                kernel_size=1,
                bias=False,
            ),  # nn.Conv2d(filters[0] + filters[1], 2 * (filters[0] + filters[1]), kernel_size=1, bias=False),
            e2nn.InnerBatchNorm(
                e2nn.FieldType(r2_act, filters[1] * 3 * [r2_act.regular_repr])
            ),  # nn.BatchNorm2d(2 * (filters[0] + filters[1])),
        )

        self.downsample_2 = nn.Sequential(
            e2nn.R2Conv(
                e2nn.FieldType(r2_act, filters[1] * 3 * [r2_act.regular_repr]),
                e2nn.FieldType(r2_act, filters[2] * 3 * [r2_act.regular_repr]),
                kernel_size=1,
                bias=False,
            ),  # nn.Conv2d(filters[1] + filters[2], 2 * (filters[1] + filters[2]), kernel_size=1, bias=False),
            e2nn.InnerBatchNorm(
                e2nn.FieldType(r2_act, filters[2] * 3 * [r2_act.regular_repr])
            ),  # nn.BatchNorm2d(2 * (filters[1] + filters[2])),
        )

        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(
            self.find_dimenstion(resolution, input_channels), dense_layers[0]
        )
        self.batchnorm4 = nn.BatchNorm1d(dense_layers[0])
        self.fc2 = nn.Linear(dense_layers[0], dense_layers[1])
        self.batchnorm5 = nn.BatchNorm1d(dense_layers[1])
        self.fc3 = nn.Linear(dense_layers[1], output_channels)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        residual = x

        # Primo blocco Inception con batchnorm e downsampling
        x = self.inception1(x)
        x = self.relu1(self.batchnorm1(x)) + self.downsample_0(residual)
        x = self.max_pool1(x)

        residual = x

        # Secondo blocco Inception
        x = self.inception2(x)
        x = self.relu2(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool2(x)

        residual = x

        # Terzo blocco Inception
        x = self.inception3(x)
        x = self.relu3(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool3(x)

        # Convertire in tensore classico per gli strati fully connected
        x = x.tensor
        x = self.flatten(x)

        # Passaggio ai layer fully connected
        x = self.fc1(x)
        x = self.relu(self.batchnorm4(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(self.batchnorm5(x))
        x = self.fc3(x)

        return x

    def find_dimenstion(self, resolution, input_channels):
        x = torch.rand(1, input_channels, resolution, resolution)

        x = e2nn.GeometricTensor(
            x, e2nn.FieldType(self.r2_act, x.shape[1] * [self.r2_act.trivial_repr])
        )

        residual = x

        # Primo blocco Inception con batchnorm e downsampling
        x = self.inception1(x)
        x = self.relu1(self.batchnorm1(x)) + self.downsample_0(residual)
        x = self.max_pool1(x)

        residual = x

        # Secondo blocco Inception

        x = self.inception2(x)
        x = self.relu2(self.batchnorm2(x)) + self.downsample_1(residual)
        x = self.max_pool2(x)

        residual = x

        # Terzo blocco Inception
        x = self.inception3(x)
        x = self.relu3(self.batchnorm3(x)) + self.downsample_2(residual)
        x = self.max_pool3(x)

        # Convertire in tensore classico per gli strati fully connected
        x = x.tensor
        x = self.flatten(x)

        return x.size()[1]


if __name__ == "__main__":
    pass
