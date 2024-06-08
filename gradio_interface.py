try:
    import gradio as gr
    from PIL import Image
    import numpy as np
    from lib.lib_trainer_predictor_lightning import MyRegressor
    import hydra
    from pathlib import Path
    import torch
    import cv2
    from lib.lib_utils import Utils
    from pytorch_lightning import seed_everything
    import os

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def get_model_names(checkpoints_path: Path):
    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    return str(best_loss[0])


def load_image(img: Image, resolution: int, enlargement_method: str = "padding"):
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (
        enlargement_method == "padding"
        and image.shape[0] < resolution
        and image.shape[1] < resolution
    ):
        image = Utils.padding_image(image, size=resolution)
    else:
        image = cv2.resize(
            image, (resolution, resolution), interpolation=cv2.INTER_CUBIC
        )

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
    image = image.to(device)

    return image


def make_prediction(model, image, target: str):
    prediction = model(image)
    prediction = torch.squeeze(prediction).detach().cpu().numpy()

    if target == "total_energy":
        n_atoms = np.loadtxt(str(Path(__file__).parent.joinpath("sample.txt")))
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


def load_model(cfg, checkpoints_path: Path):
    checkpoints = get_model_names(checkpoints_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyRegressor(cfg)
    model.load_state_dict(torch.load(checkpoints, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    return model


def read_from_xyz_file(xyz_file: gr.File):
    """Read xyz files and return lists of x,y,z coordinates and atoms"""

    X = []
    Y = []
    Z = []
    atoms = []

    with open(xyz_file.name, "r") as f:
        for line in f:
            l = line.split()
            if len(l) == 4 or len(l) == 5:
                X.append(float(l[1]))
                Y.append(float(l[2]))
                Z.append(float(l[3]))
                atoms.append(str(l[0]))

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    return X, Y, Z, atoms


def generate_num_atoms(
    xyz_file: gr.File,
):
    X, Y, Z, atoms = read_from_xyz_file(xyz_file)

    n_C = atoms.count("C")
    n_O = atoms.count("O")
    n_H = atoms.count("H")

    lines = []
    with open(Path(__file__).parent.joinpath("sample.txt"), "w") as f:
        lines.append(f"{n_C}\n") if n_C > 0 else None
        lines.append(f"{n_O}\n") if n_O > 0 else None
        lines.append(f"{n_H}\n") if n_H > 0 else None

        f.writelines(lines)

    return len(lines)


def crop_image(
    image: Image,
) -> Image:
    image_data = np.asarray(image)
    if len(image_data.shape) == 2:
        image_data_bw = image_data
    else:
        image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    cropBox = (
        min(non_empty_rows),
        max(non_empty_rows),
        min(non_empty_columns),
        max(non_empty_columns),
    )

    if len(image_data.shape) == 2:
        image_data_new = image_data[
            cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1
        ]
    else:
        image_data_new = image_data[
            cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :
        ]

    new_image = Image.fromarray(image_data_new)

    return new_image


def generate_png(xyz_file: gr.File):
    X, Y, Z, atoms = read_from_xyz_file(xyz_file)

    z_max = np.max(Z)
    z_min = np.min(Z)

    if z_max == z_min == 0.0:
        z_min = -1.0
        z_max = 0.0

    x_max = np.max(X)
    x_min = np.min(X)

    y_max = np.max(Y)
    y_min = np.max(Y)

    resolution = round(
        4 * (5 + np.max([np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]))
    )

    C = np.zeros((resolution, resolution))
    O = np.zeros((resolution, resolution))
    H = np.zeros((resolution, resolution))

    z_norm = lambda x: (x - z_min) / (z_max - z_min)

    for i in range(len(X)):
        if atoms[i] == "C":
            x_coord = int(round(X[i] * 2) + resolution / 2)
            y_coord = int(round(Y[i] * 2) + resolution / 2)
            if C[y_coord, x_coord] < z_norm(Z[i]):
                C[y_coord, x_coord] = z_norm(Z[i])
        elif atoms[i] == "O":
            x_coord = int(round(X[i] * 2) + resolution / 2)
            y_coord = int(round(Y[i] * 2) + resolution / 2)
            if O[y_coord, x_coord] < z_norm(Z[i]):
                O[y_coord, x_coord] = z_norm(Z[i])
        elif atoms[i] == "H":
            x_coord = int(round(X[i] * 2) + resolution / 2)
            y_coord = int(round(Y[i] * 2) + resolution / 2)
            if H[y_coord, x_coord] < z_norm(Z[i]):
                H[y_coord, x_coord] = z_norm(Z[i])

    Matrix = np.stack((C, O, H), axis=2)
    Matrix = (Matrix * 255.0).astype(np.uint8)

    image = Image.fromarray(Matrix)
    try:
        new_image = crop_image(image)
    except:
        new_image = image.copy()

    return new_image


def xyz_interface(xyz_file, target):
    image = generate_png(xyz_file)
    image.save("/home/tommaso/git_workspace/GrapheNet/gradio/test.png")
    num_different_atoms = generate_num_atoms(xyz_file)
    prediction = None

    match target:
        case "Electron Affinity":
            target = "electron_affinity"
        case "Electronegativity":
            target = "electronegativity"
        case "Ionization Potential":
            target = "ionization_potential"
        case "Fermi Energy":
            target = "Fermi_energy"
        case "Total Energy":
            target = "total_energy"

    @hydra.main(version_base="1.2", config_path="config", config_name="gradio")
    def main(cfg):
        cfg.target = target

        seed_everything(42, workers=True)

        model = load_model(
            cfg,
            checkpoints_path=Path(__file__).parent.joinpath(
                "gradio", "models", f"{target}"
            ),
        )

        img = load_image(image, cfg.resolution, cfg.enlargement_method)

        nonlocal prediction
        prediction = make_prediction(model, img, target=target)

    main()
    os.remove(str(Path(__file__).parent.joinpath("sample.txt")))
    return image, f"The predicted value for {target} is equal to: {prediction:.4f}"


if __name__ == "__main__":
    # Definisci le opzioni per il menu a tendina
    dropdown_options = [
        "Electron Affinity",
        "Electronegativity",
        "Ionization Potential",
        "Fermi Energy",
        "Total Energy",
    ]

    # Crea il componente Dropdown
    dropdown = gr.components.Dropdown(
        choices=dropdown_options, label="Select the target to predict"
    )

    # Definisci l'interfaccia utilizzando Gradio
    iface = gr.Interface(
        fn=xyz_interface, inputs=["file", dropdown], outputs=["image", "text"]
    )

    # Avvia l'interfaccia
    iface.launch(share=True)
