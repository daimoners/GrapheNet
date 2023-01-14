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


def plot_feature_maps(conv_layers, image, save_path: Path, num_filters: int = 10):

    # pass the image through all the layers
    results = [conv_layers[0](image)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    feature_maps_path = save_path.joinpath("feature_maps")
    feature_maps_path.mkdir(parents=True, exist_ok=True)

    layers = {0: 64, 1: 128, 2: 128, 3: 256, 4: 256}
    # visualize "num_filters" features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(5, 5))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        for i, filter in enumerate(layer_viz):
            if i == num_filters:  # we will visualize only 8x8 blocks from each layer
                break
            plt.imshow(filter.detach().cpu(), cmap="inferno")
            plt.savefig(
                str(
                    feature_maps_path.joinpath(
                        f"layer_{layers[num_layer]}_filter_{i}.png"
                    )
                )
            )
        plt.close()


def load_image(img_path: Path, resolution: int):

    image = cv2.imread(str(img_path), -1)
    image = Utils.padding_image(image, size=resolution)
    image = np.asarray(image, float) / 255.0
    image = torch.unsqueeze(
        torch.from_numpy(np.expand_dims(image.copy(), 0)).float(), 0
    )
    image = image.to(torch.device("cuda"))

    return image


def make_prediction(model, image, sample_path: Path, target: str = "total_energy"):
    prediction = model(image)
    prediction = torch.squeeze(prediction).detach().cpu().numpy()

    if target == "total_energy":
        n_atoms = np.loadtxt(str(sample_path.with_suffix(".txt")))
        prediction = prediction[0] + n_atoms * prediction[1]

    return prediction


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
    save_path = Path(cfg.package_path).joinpath("single_prediction")
    save_path.mkdir(exist_ok=True, parents=True)

    model = load_model(cfg, checkpoints_path=Path(cfg.package_path).joinpath("models"))

    model_weights, conv_layers = get_weights_and_layers(model)

    image = load_image(Path(cfg.sample), cfg.resolution)

    prediction = make_prediction(model, image, Path(cfg.sample))
    print(f"Predicted target: {prediction:.4f}")

    plot_filters(model_weights, save_path)
    plot_feature_maps(conv_layers, image, save_path)


if __name__ == "__main__":
    main()
