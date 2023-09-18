try:

    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from pytorch_lightning import Trainer, seed_everything
    from pathlib import Path
    import numpy as np
    from lib.lib_utils import Utils
    import torch.nn as nn
    import torch
    import cv2
    import matplotlib.pyplot as plt
    import shutil
    import pandas as pd
    from tqdm import tqdm

except Exception as e:

    print("Some module are missing {}".format(e))


def get_model_names(checkpoints_path: Path):

    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    return str(best_loss[0])


def plot_filters(model_weights, save_path: Path):

    filters_path = save_path.joinpath("filters")
    filters_path.mkdir(parents=True, exist_ok=True)

    print("Plotting filters...")

    plt.figure(figsize=(15, 15))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(
            8, 8, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(str(filters_path.joinpath("filter_64.png")))
    plt.close()

    plt.figure(figsize=(15, 15))
    for i, filter in enumerate(model_weights[1]):
        plt.subplot(
            12, 12, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(str(filters_path.joinpath("filter_128_1.png")))
    plt.close()

    plt.figure(figsize=(15, 15))
    for i, filter in enumerate(model_weights[2]):
        plt.subplot(
            12, 12, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(str(filters_path.joinpath("filter_128_2.png")))
    plt.close()

    plt.figure(figsize=(15, 15))
    for i, filter in enumerate(model_weights[3]):
        plt.subplot(
            16, 16, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(str(filters_path.joinpath("filter_256_1.png")))
    plt.close()

    plt.figure(figsize=(15, 15))
    for i, filter in enumerate(model_weights[4]):
        plt.subplot(
            16, 16, i + 1
        )  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(str(filters_path.joinpath("filter_256_2.png")))
    plt.close()


def inception_plot_filters(model_weights, save_path: Path):

    filters_path = save_path.joinpath("filters")
    filters_path.mkdir(parents=True, exist_ok=True)

    print("Plotting filters...")
    layers = {
        0: "conv1_32",
        1: "conv3_32",
        2: "conv5_32",
        3: "conv1_64",
        4: "conv3_64",
        5: "conv5_64",
        6: "conv1_128",
        7: "conv3_128",
        8: "conv5_128",
    }
    for k in range(0, len(model_weights)):
        plt.figure(figsize=(15, 15))
        for i, filter in enumerate(model_weights[k]):
            if k in [0, 1, 2]:
                plt.subplot(6, 6, i + 1)
            elif k in [3, 4, 5]:
                plt.subplot(8, 8, i + 1)
            elif k in [6, 7, 8]:
                plt.subplot(12, 12, i + 1)
            plt.imshow(filter[0, :, :].detach().cpu(), cmap="gray")
            plt.axis("off")
            plt.savefig(str(filters_path.joinpath(f"filter_{layers[k]}.png")))
        plt.close()


def plot_feature_maps(conv_layers, image, save_path: Path, num_filters: int = 10):

    print("Plotting feature maps...")

    # pass the image through all the layers
    results = [conv_layers[0](image)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    feature_maps_path = save_path.joinpath("feature_maps")
    feature_maps_path.mkdir(parents=True, exist_ok=True)

    layers = {0: "64", 1: "128_1", 2: "128_2", 3: "256_1", 4: "256_2"}
    # visualize "num_filters" features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(5, 5))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == num_filters:  # we will visualize only 8x8 blocks from each layer
                break
            plt.imshow(filter.detach().cpu(), cmap="plasma")
            plt.savefig(
                str(
                    feature_maps_path.joinpath(
                        f"layer_{layers[num_layer]}_filter_{i}.png"
                    )
                )
            )
        plt.close()


def inception_plot_feature_maps(
    conv_layers, image, save_path: Path, num_filters: int = 10, colormap="seismic"
):

    print("Plotting feature maps...")

    # pass the image through all the layers
    results1 = [conv_layers[0](image)]
    results2 = [conv_layers[1](image)]
    results3 = [conv_layers[2](image)]

    for i in range(0, int(len(conv_layers) / 3) - 1):
        # pass the result from the last layer to the next layer
        complete_result = torch.cat([results1[-1], results2[-1], results3[-1]], dim=1)

        results1.append(conv_layers[i * (i + 2) + 3](complete_result))
        results2.append(conv_layers[i * (i + 2) + 4](complete_result))
        results3.append(conv_layers[i * (i + 2) + 5](complete_result))
    # make a copy of the `results`
    outputs = [*results1, *results2, *results3]

    feature_maps_path = save_path.joinpath("feature_maps")
    feature_maps_path.mkdir(parents=True, exist_ok=True)

    layers = {
        0: "conv1_32",
        1: "conv3_32",
        2: "conv5_32",
        3: "conv1_64",
        4: "conv3_64",
        5: "conv5_64",
        6: "conv1_128",
        7: "conv3_128",
        8: "conv5_128",
    }
    # visualize "num_filters" features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in tqdm(range(len(outputs))):
        plt.figure(figsize=(5, 5))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if num_filters == -1:
                pass
            elif i == num_filters:  # we will visualize only 8x8 blocks from each layer
                break
            plt.imshow(filter.detach().cpu(), cmap=colormap)
            plt.savefig(
                str(
                    feature_maps_path.joinpath(
                        f"layer_{layers[num_layer]}_filter_{i}.png"
                    )
                )
            )
        plt.close()


def load_image(img_path: Path, resolution: int, enlargement_method: str = "padding"):

    image = cv2.imread(str(img_path), -1)
    if enlargement_method == "padding":
        image = Utils.padding_image(image, size=resolution)
    elif enlargement_method == "resize":
        image = cv2.resize(
            image, (resolution, resolution), interpolation=cv2.INTER_CUBIC
        )
    else:
        raise Exception("Wrong enlargement method!")

    image = np.asarray(image, float) / 255.0
    if len(image.shape) == 2:
        image = torch.unsqueeze(
            torch.from_numpy(np.expand_dims(image.copy(), 0)).float(), 0
        )
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = (
            torch.from_numpy(np.expand_dims(image.copy(), 0))
            .permute(0, 3, 1, 2)
            .float()
        )
    else:
        raise Exception("Wrong dimensions for the input images\n")
    image = image.to(torch.device("cuda"))

    return image


def make_prediction(model, image, sample_path: Path, target: str):
    prediction = model(image)
    prediction = torch.squeeze(prediction).detach().cpu().numpy()

    if target == "total_energy":
        n_atoms = np.loadtxt(str(sample_path.with_suffix(".txt")))
        if prediction.size == 2:
            prediction = prediction[0] + n_atoms * prediction[1]
        elif prediction.size == 4:
            prediction = (
                prediction[0]
                + n_atoms[0] * prediction[1]
                + n_atoms[1] * prediction[2]
                + n_atoms[2] * prediction[3]
            )

    return prediction


def get_target_value(sample_name: str, target: str, csv_path: Path):
    df = pd.read_csv(csv_path)

    df_filtered = df.loc[df["file_name"] == sample_name]

    # Extract the value of column 'A' for the filtered rows
    value = df_filtered[target].iloc[0]

    return value


def get_weights_and_layers(model):
    model_weights = []
    conv_layers = []
    model_children = list(model.net.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
    print(f"Total convolution layers: {counter}")

    return model_weights, conv_layers


def new_get_weights_and_layers(model):
    model_weights = []
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "down" not in name:
            model_weights.append(module.weight)
            conv_layers.append(module)
    print(f"Total convolution layers: {len(conv_layers)}")

    return model_weights, conv_layers


def load_model(cfg, checkpoints_path: Path):
    checkpoints = get_model_names(checkpoints_path)

    model = MyRegressor(cfg)
    model.load_state_dict(torch.load(checkpoints)["state_dict"])
    model.to(torch.device("cuda"))
    model.eval()

    return model


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):

    seed_everything(42, workers=True)
    save_path = Path(cfg.train.spath).joinpath(
        "models", f"{cfg.target}", f"visualization_{Path(cfg.sample).stem}"
    )
    save_path.mkdir(exist_ok=True, parents=True)

    model = load_model(
        cfg, checkpoints_path=Path(cfg.train.spath).joinpath("models", f"{cfg.target}")
    )

    model_weights, conv_layers = new_get_weights_and_layers(model)

    image = load_image(Path(cfg.sample), cfg.resolution, cfg.enlargement_method)

    prediction = make_prediction(model, image, Path(cfg.sample), target=cfg.target)
    print(f"Predicted target: {prediction:.4f}")
    target_value = get_target_value(
        Path(cfg.sample).stem, cfg.target, Path(cfg.train.spath).joinpath("dataset.csv")
    )
    print(f"Real target: {target_value:.4f}")
    print(f"MAE: {abs((prediction-target_value)/target_value)*100.0:.6f}%")

    shutil.copy(cfg.sample, str(save_path))
    inception_plot_feature_maps(conv_layers, image, save_path, num_filters=-1)
    # inception_plot_filters(model_weights, save_path)


if __name__ == "__main__":
    main()
