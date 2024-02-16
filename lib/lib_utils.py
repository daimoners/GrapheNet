try:
    import numpy as np
    from datetime import datetime
    import seaborn as sns
    import pandas as pd
    import shutil
    from tqdm import tqdm
    import random
    from PIL import Image
    import cv2
    from time import time
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    from pathlib import Path
    import matplotlib.pyplot as plt
    import math
    from scipy.stats import boxcox
    import yaml
    from sklearn.model_selection import train_test_split
    from scipy.spatial.distance import pdist

except Exception as e:
    print("Some module are missing {}".format(e))


class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    @staticmethod
    def find_first_empty_cell(matrix, x, y):
        rows, cols = matrix.shape
        adjacent_cells = [
            (x - 1, y - 1),
            (x - 1, y),
            (x - 1, y + 1),
            (x, y - 1),
            (x, y + 1),
            (x + 1, y - 1),
            (x + 1, y),
            (x + 1, y + 1),
        ]

        for adj_x, adj_y in adjacent_cells:
            if 0 <= adj_x < rows and 0 <= adj_y < cols:
                if matrix[adj_y, adj_x] == 0:
                    return adj_x, adj_y

        return None

    @staticmethod
    def read_from_xyz_file(spath: Path):
        """Read xyz files and return lists of x,y,z coordinates and atoms"""

        X = []
        Y = []
        Z = []
        atoms = []

        with open(str(spath), "r") as f:
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

    @staticmethod
    def crop_image(image: Image, name: str = None, dpath: Path = None) -> Image:
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
        if dpath is not None:
            new_image.save(dpath.joinpath(name))

        return new_image

    @staticmethod
    def generate_png(
        spath: Path,
        dpath: Path,
        z_relative=False,
        single_channel_images=False,
    ):
        """Generate a .npy matrix starting from lists of x,y,z coordinates"""

        X, Y, Z, atoms = Utils.read_from_xyz_file(spath)

        if z_relative:
            z_max = np.max(Z)
            z_min = np.min(Z)

            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )
        else:
            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            z_max = x[0][2]
            z_min = x[1][2]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )

        C = np.zeros((resolution, resolution))
        O = np.zeros((resolution, resolution))
        H = np.zeros((resolution, resolution))

        z_norm = lambda x: (x - z_min) / (z_max - z_min)

        C_only = True

        count = 0

        for i in range(len(X)):
            if atoms[i] == "C":
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if C[y_coord, x_coord] < z_norm(Z[i]):
                    C[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "O":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if O[y_coord, x_coord] < z_norm(Z[i]):
                    O[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "H":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if H[y_coord, x_coord] < z_norm(Z[i]):
                    H[y_coord, x_coord] = z_norm(Z[i])

        name = spath.stem

        if single_channel_images:
            C = (C * 255.0).astype(np.uint8)
            O = (O * 255.0).astype(np.uint8)
            H = (H * 255.0).astype(np.uint8)

            image_C = Image.fromarray(C)
            Utils.crop_image(image_C, name + "_C.png", dpath)
            image_O = Image.fromarray(O)
            Utils.crop_image(image_O, name + "_O.png", dpath)
            image_H = Image.fromarray(H)
            Utils.crop_image(image_H, name + "_H.png", dpath)

        else:
            if C_only:
                Matrix = C.copy()
            else:
                Matrix = np.stack((C, O, H), axis=2)
            Matrix = (Matrix * 255.0).astype(np.uint8)
            # Matrix = np.flip(Matrix, 0)

            image = Image.fromarray(Matrix)
            Utils.crop_image(image, name + ".png", dpath)

            return count

    @staticmethod
    def generate_grayscale_png(
        spath: Path,
        dpath: Path,
        z_relative=False,
    ):
        """Generate a .npy matrix starting from lists of x,y,z coordinates"""

        X, Y, Z, atoms = Utils.read_from_xyz_file(spath)

        if z_relative:
            z_max = np.max(Z)
            z_min = np.min(Z)

            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )
        else:
            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            z_max = x[0][2]
            z_min = x[1][2]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )

        M = np.zeros((resolution, resolution))

        z_norm = lambda x: (x - z_min) / (z_max - z_min)

        C_only = True

        count = 0

        for i in range(len(X)):
            x_coord = int(round(X[i] * 2) + resolution / 2)
            y_coord = int(round(Y[i] * 2) + resolution / 2)
            if M[y_coord, x_coord] < z_norm(Z[i]):
                M[y_coord, x_coord] = z_norm(Z[i])

        name = spath.stem

        Matrix = (M * 255.0).astype(np.uint8)
        # Matrix = np.flip(Matrix, 0)

        image = Image.fromarray(Matrix)
        Utils.crop_image(image, name + ".png", dpath)

        return count

    @staticmethod
    def dataset_max_and_min(spath: Path, dpath: Path = None) -> list:
        """
        This static method returns a list of the maximum and minimum values for each coordinate given a folder of .xyz files. It takes two parameters, spath (Path) and dpath (Path), with dpath being optional. It creates two lists, MAX and MIN, which are initialized to [0, 0, 0]. It then iterates through the files in the spath directory and checks if each file is an .xyz file. If it is, it calls the find_max_and_min() method from Utils to get the max and min values for that file. It then compares these values to MAX and MIN respectively to update them if necessary. Finally, if dpath is not None, it saves the MAX and MIN lists as a text file in the dpath directory. Otherwise it just prints out MAX and MIN. The method returns both MAX and MIN as a list.
        """
        """Return a list of Max and Min for each coordinate, given a folder of .xyz files"""

        MAX = [0, 0, 0]
        MIN = [0, 0, 0]

        max = []
        min = []

        for file in spath.iterdir():
            if file.suffix == ".xyz":
                max, min = Utils.find_max_and_min(file)
                if max[0] > MAX[0]:
                    MAX[0] = max[0]
                if min[0] < MIN[0]:
                    MIN[0] = min[0]
                if max[1] > MAX[1]:
                    MAX[1] = max[1]
                if min[1] < MIN[1]:
                    MIN[1] = min[1]
                if max[2] > MAX[2]:
                    MAX[2] = max[2]
                if min[2] < MIN[2]:
                    MIN[2] = min[2]

        if dpath is not None:
            np.savetxt(dpath.joinpath("max_min_coordinates.txt"), [MAX, MIN])
        else:
            print([MAX, MIN])

        return MAX, MIN

    @staticmethod
    def find_max_and_min(spath: Path):
        """Return a list of Max and Min for each coordinate, given a single .xyz file"""

        X = []
        Y = []
        Z = []

        with open(str(spath), "r") as f:
            for line in f:
                l = line.split()
                if len(l) == 4 or len(l) == 5:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        max = [np.max(X), np.max(Y), np.max(Z)]
        min = [np.min(X), np.min(Y), np.min(Z)]

        return max, min

    @staticmethod
    def train_val_test_split_png(
        spath: Path,
        dpath: Path,
        csv: Path,
        features: list,
        split: list = 0.7,
        suffix: str = ".png",
    ):
        """Split a dataset in test and train and generate the respective CSV files"""

        train_path = dpath.joinpath("train")
        test_path = dpath.joinpath("test")
        val_path = dpath.joinpath("val")

        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv)

        X = np.zeros(len(df))

        y = df[features[0]]

        # Split the data into training and test sets
        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_indices,
            test_indices,
        ) = train_test_split(X, y, df.index, test_size=1 - split, random_state=42)

        # Split the test set into validation and test sets
        X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(
            X_test, y_test, test_indices, test_size=0.5, random_state=42
        )

        # Start moving train sample

        train_csv = df.loc[train_indices, features]
        names = train_csv["file_name"].tolist()
        for name in tqdm(names):
            samples = [
                f
                for f in spath.iterdir()
                if f.stem == name or f.stem.startswith(f"{name}_R")
            ]
            for sample in samples:
                shutil.copy(sample, train_path.joinpath(f"{sample.stem}{suffix}"))
        train_csv.to_csv(train_path.joinpath("train.csv"))

        print("Train files moved\n")

        # Start moving validation sample

        val_csv = df.loc[val_indices, features]
        names = val_csv["file_name"].tolist()
        for name in tqdm(names):
            samples = [
                f
                for f in spath.iterdir()
                if f.stem == name or f.stem.startswith(f"{name}_R")
            ]
            for sample in samples:
                shutil.copy(sample, val_path.joinpath(f"{sample.stem}{suffix}"))
        val_csv.to_csv(val_path.joinpath("val.csv"))

        print("Validation files moved\n")

        # Start moving test sample

        test_csv = df.loc[test_indices, features]
        names = test_csv["file_name"].tolist()
        for name in tqdm(names):
            samples = [
                f
                for f in spath.iterdir()
                if f.stem == name or f.stem.startswith(f"{name}_R")
            ]
            for sample in samples:
                shutil.copy(sample, test_path.joinpath(f"{sample.stem}{suffix}"))
        test_csv.to_csv(test_path.joinpath("test.csv"))

        print("Test files moved\n")

        df = df.loc[:, features]
        df.to_csv(dpath.joinpath("dataset.csv"))

    @staticmethod
    def find_max_dimensions_png_folder(spath: Path, dpath: Path = None):
        """Find the maximum dimensions in a folder of images"""
        heights = []
        widths = []

        for file in spath.iterdir():
            if file.suffix in Utils.IMAGE_EXTENSIONS:
                width, height = Image.open(file).size
                heights.append(height)
                widths.append(width)

        heights = np.asarray(heights)
        widths = np.asarray(widths)

        if dpath is None:
            return np.max(widths), np.max(heights)
        else:
            np.savetxt(
                dpath.joinpath("max_images_dimensions.txt"),
                [np.max(widths), np.max(heights)],
            )

    @staticmethod
    def padding_image(image, size=160):
        h = image.shape[0]
        w = image.shape[1]

        top = round((size - h) / 2)
        bottom = size - (h + top)

        left = round((size - w) / 2)
        right = size - (w + left)

        padded_img = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0
        )

        return padded_img

    @staticmethod
    def build_csv_from_png_files(spath: Path, targets: list = ["total_energy"]):
        train_path = spath.joinpath("train")
        test_path = spath.joinpath("test")
        val_path = spath.joinpath("val")
        dataset_csv_path = spath.joinpath("dataset.csv")

        all_csv = pd.read_csv(dataset_csv_path)

        indices = []

        train_items = len(
            [
                f
                for f in train_path.iterdir()
                if f.suffix in Utils.IMAGE_EXTENSIONS and f.is_file()
            ]
        )
        pbar = tqdm(total=train_items)
        for file in train_path.iterdir():
            if file.suffix in Utils.IMAGE_EXTENSIONS:
                index = all_csv.index[all_csv["file_name"] == file.stem].tolist()
                if len(index) != 0:
                    indices.append(int(index[0]))
                    pbar.update(1)
        pbar.close()

        new_csv = all_csv.loc[all_csv.index[indices], ["file_name", *targets]]
        new_csv.to_csv(train_path.joinpath("train.csv"))

        test_items = len(
            [
                f
                for f in test_path.iterdir()
                if f.suffix in Utils.IMAGE_EXTENSIONS and f.is_file()
            ]
        )
        pbar = tqdm(total=test_items)
        indices.clear()
        for file in test_path.iterdir():
            if file.suffix in Utils.IMAGE_EXTENSIONS:
                index = all_csv.index[all_csv["file_name"] == file.stem].tolist()
                if len(index) != 0:
                    indices.append(int(index[0]))
                    pbar.update(1)
        pbar.close()

        new_csv = all_csv.loc[all_csv.index[indices], ["file_name", *targets]]
        new_csv.to_csv(test_path.joinpath("test.csv"))

        val_items = len(
            [
                f
                for f in val_path.iterdir()
                if f.suffix in Utils.IMAGE_EXTENSIONS and f.is_file()
            ]
        )
        pbar = tqdm(total=val_items)
        indices.clear()
        for file in val_path.iterdir():
            if file.suffix in Utils.IMAGE_EXTENSIONS:
                index = all_csv.index[all_csv["file_name"] == file.stem].tolist()
                if len(index) != 0:
                    indices.append(int(index[0]))
                    pbar.update(1)
        pbar.close()

        new_csv = all_csv.loc[all_csv.index[indices], ["file_name", *targets]]
        new_csv.to_csv(val_path.joinpath("val.csv"))

    @staticmethod
    def find_num_workers(dataset):
        seconds = []
        workers = []

        for num_workers in range(2, mp.cpu_count(), 2):
            train_loader = DataLoader(
                dataset,
                shuffle=True,
                num_workers=num_workers,
                batch_size=64,
                pin_memory=True,
            )
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time()

            seconds.append(end - start)
            workers.append(num_workers)

        index_min = seconds.index(min(seconds))

        return workers[index_min]

    @staticmethod
    def plot_distribution(spath: Path, csv: Path, features: list, dpath: Path):
        dpath.mkdir(parents=True, exist_ok=True)

        sns.set_style("white")

        colors = ["dodgerblue", "deeppink", "gold", "green", "red", "purple"]

        # Import data
        df = pd.read_csv(csv)

        indices = []
        for f in spath.iterdir():
            if f.suffix in Utils.IMAGE_EXTENSIONS:
                index = df.index[df["file_name"] == f.stem].tolist()
                if len(index) != 0:
                    indices.append(index[0])

        for i in range(len(features)):
            x = df[features[i]][indices]

            # Plot
            kwargs = dict(hist_kws={"alpha": 0.6}, kde_kws={"linewidth": 2})

            plt.figure(figsize=(15, 10), dpi=100)
            try:
                sns.distplot(x, color=colors[i], label=features[i], **kwargs)
            except:
                pass
            plt.legend()
            plt.savefig(dpath.joinpath(features[i] + ".png"))

    @staticmethod
    def date_and_time() -> str:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        return dt_string

    @staticmethod
    def compute_min_max_target(
        csv_path: Path, target="total_energy", all_positive=False
    ):
        dataset = pd.read_csv(str(csv_path))

        target_array = (
            np.abs(np.asarray(dataset[target]))
            if all_positive
            else np.asarray(dataset[target])
        )

        np.savetxt(
            str(csv_path.parent.joinpath(f"min_max_{target}.txt")),
            [np.min(target_array), np.max(target_array)],
        )

        print(f"Path: {csv_path}")
        print(f"Min: {np.min(target_array)}")
        print(f"Max: {np.max(target_array)}")

    @staticmethod
    def compute_mean_std_target(
        csv_path: Path, target="total_energy", all_positive=False
    ):
        dataset = pd.read_csv(str(csv_path))

        target_array = (
            np.abs(np.asarray(dataset[target]))
            if all_positive
            else np.asarray(dataset[target])
        )

        np.savetxt(
            str(csv_path.parent.joinpath(f"mean_std_{target}.txt")),
            [np.mean(target_array), np.std(target_array)],
        )

        print(f"Path: {csv_path}")
        print(f"Mean: {np.mean(target_array)}")
        print(f"STD: {np.std(target_array)}")

    @staticmethod
    def compute_minimum_target(
        csv_path: Path, target="total_energy", all_positive=False
    ):
        dataset = pd.read_csv(str(csv_path))

        target_array = np.asarray(dataset[target])

        np.savetxt(
            str(csv_path.parent.joinpath(f"minimum_{target}.txt")),
            [np.min(target_array)],
        )

        print(f"Path: {csv_path}")
        print(f"Minimum: {np.min(target_array)}")

    @staticmethod
    def compute_lambda_boxcox(csv_path: Path, target: str):
        df = pd.read_csv(csv_path)

        target_values = df[target].to_numpy()

        target_values = target_values + np.abs(np.min(target_values)) + 1

        # Apply Box-Cox transformation to the data
        transformed_data, lambda_value = boxcox(target_values)

        np.savetxt(
            str(csv_path.parent.joinpath(f"lambda_{target}.txt")),
            [lambda_value],
        )

        # Print the lambda value
        print("Lambda value:", lambda_value)
        # Print the first 5 values of the transformed data
        print("Transformed data:", transformed_data[:5])

    @staticmethod
    def generate_num_atoms(
        dataset_path: Path,
        xyz_path: Path,
        format: str = ".png",
        info_max_atoms: bool = True,
    ):
        """Generate the number and type of atoms in a .txt file for each .xyz file

        Args:
            dataset_path (Path): path of the dataset already splitted in train/val/test
            xyz_path (Path): path of all the .xyz files
            format (str, optional): image format. Defaults to ".png".
        """
        max_atoms = 0
        flake_max_atoms = ""

        for dir in ["train", "val", "test"]:
            for file in tqdm(dataset_path.joinpath(dir).iterdir()):
                if file.suffix == format and not "_R" in file.stem:
                    X, Y, Z, atoms = Utils.read_from_xyz_file(
                        xyz_path.joinpath(file.stem + ".xyz")
                    )

                    n_C = atoms.count("C")
                    n_O = atoms.count("O")
                    n_H = atoms.count("H")

                    if (n_C + n_O + n_H) > max_atoms and info_max_atoms:
                        max_atoms = n_C + n_O + n_H
                        flake_max_atoms = file.stem

                    lines = []
                    with open(dataset_path.joinpath(dir, file.stem + ".txt"), "w") as f:
                        lines.append(f"{n_C}\n") if n_C > 0 else None
                        lines.append(f"{n_O}\n") if n_O > 0 else None
                        lines.append(f"{n_H}\n") if n_H > 0 else None

                        f.writelines(lines)

        if info_max_atoms:
            lines.clear()
            with open(dataset_path.joinpath("flake_max_atoms.txt"), "w") as f:
                lines.append(f"Flake Name = {flake_max_atoms}\n")
                lines.append(f"Max Atoms = {max_atoms}\n")

                f.writelines(lines)

    @staticmethod
    def plot_fit(
        y: list, y_hat: list, dpath: Path, target: str, colormap: str = "plasma"
    ):
        def r2_score(y_pred, y_true):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            mean_y = np.mean(y_true)
            SSR = np.sum((y_pred - y_true) ** 2)
            SST = np.sum((y_true - mean_y) ** 2)
            return 1 - SSR / SST

        min = np.min([np.min(y), np.min(y_hat)])
        max = np.max([np.max(y), np.max(y_hat)])

        MAEs = np.abs(np.array(y_hat) - np.array(y)) / np.abs(np.array(y))

        plt.figure(figsize=(10, 7))
        plt.plot(
            [min, max],
            [min, max],
        )
        plt.scatter(y_hat, y, c=MAEs, cmap=plt.cm.get_cmap(colormap))
        cbar = plt.colorbar()
        cbar.set_label("Color Values")
        plt.xlabel("Predictions")
        plt.ylabel("Targets")
        plt.title(f"{target} - R2 = {r2_score(y_hat,y):.3f}")
        plt.savefig(str(dpath))

    @staticmethod
    def write_csv_results(
        y: list,
        y_hat: list,
        names: list,
        dpath: Path,
        target: str,
    ):
        df = pd.DataFrame()
        df["file_name"] = names
        df[f"{target}_real"] = y
        df[f"{target}_predicted"] = y_hat
        df[f"{target}_MAE"] = (
            np.abs(np.array(y_hat) - np.array(y)) / np.abs(np.array(y)) * 100.0
        )

        df = df.sort_values(by=f"{target}_MAE", ascending=False)

        df.to_csv(dpath)

    @staticmethod
    def drop_custom(
        df: pd.DataFrame,
    ):
        print("Dropping outliers from the dataset...\n")

        # el_aff_down = df.index[df["electron_affinity"] <= -7].tolist()
        el_aff_up = df.index[df["electron_affinity"] > -5.7].tolist()
        el_aff = [*el_aff_up]

        # elneg_down = df.index[df["electronegativity"] <= -6.2].tolist()
        # elneg_up = df.index[df["electronegativity"] >= -4].tolist()
        # elneg = [*elneg_down, *elneg_up]

        # i_pot_down = df.index[df["ionization_potential"] <= -6.0].tolist()
        # i_pot_up = df.index[df["ionization_potential"] >= -3.5].tolist()
        # i_pot = [*i_pot_down, *i_pot_up]

        # indices = [*el_aff, *elneg, *i_pot]

        indices = [*el_aff]

        df = df.drop(indices, axis=0)

        return df

    @staticmethod
    def drop_by_oxygen_distribution(
        csv_path: Path,
        xyz_path: Path,
        threshold: float = 0.15,
        targets: list = [
            "electronegativity",
            "total_energy",
            "electron_affinity",
            "ionization_potential",
            "Fermi_energy",
        ],
    ) -> pd.DataFrame:
        MAX, MIN = OxygenUtils.find_max_min_distribution(csv_path, xyz_path, targets)

        df = pd.read_csv(str(csv_path))

        df = Utils.drop_nan_and_zeros(df)

        names = df["file_name"].tolist()

        distribution = []
        d = []

        for file in tqdm(names):
            X, Y, Z, atoms = Utils.read_from_xyz_file(xyz_path.joinpath(file + ".xyz"))

            distribution.clear()

            for i in range(len(X)):
                if atoms[i] == "O":
                    d.clear()
                    for j in range(len(X)):
                        if atoms[j] == "O" and i != j:
                            P1 = [X[i], Y[i]]
                            P2 = [X[j], Y[j]]
                            d.append(math.dist(P1, P2))

                    distribution.append((np.mean(d) - MIN) / (MAX - MIN))

            if np.mean(distribution) <= threshold:
                names.remove(file)

        print(f"Lenght of the dataset after dropping the outliers: {len(names)}")

        df = df[df["file_name"].isin(names)]

        return df

    @staticmethod
    def drop_nan_and_zeros(
        df: pd.DataFrame,
        targets: list = [
            "electronegativity",
            "total_energy",
            "electron_affinity",
            "ionization_potential",
            "Fermi_energy",
            "band_gap",
        ],
    ) -> pd.DataFrame:
        df = df.dropna(subset=targets)

        indices = []
        for t in targets:
            idx = df.index[df[t] == 0.0].tolist()
            indices = [*indices, *idx]
        df.drop(indices, axis=0, inplace=True)

        return df

    @staticmethod
    def create_subset_xyz(
        xyz_path: Path,
        dpath: Path,
        n_items: int,
        targets=[
            "electronegativity",
            "total_energy",
            "electron_affinity",
            "ionization_potential",
            "Fermi_energy",
        ],
        oxygen_distribution_threshold: float | None = None,
        drop_custom: bool = False,
        min_num_atoms: int | list = None,
    ):
        targets = ["file_name", *targets]

        dpath.mkdir(parents=True, exist_ok=True)

        if (
            oxygen_distribution_threshold is None
            or oxygen_distribution_threshold == 0.0
        ):
            df = pd.read_csv(xyz_path.joinpath("dataset.csv"))

            df = Utils.drop_nan_and_zeros(df)

            if drop_custom:
                df = Utils.drop_custom(df)

            if min_num_atoms is not None or min_num_atoms != 0:
                df = Utils.drop_min_num_atoms(df, min_num_atoms)

            names = df["file_name"].tolist()

            items = []

            if n_items == 0:
                for file in tqdm(names):
                    shutil.copy(
                        xyz_path.joinpath(file + ".xyz"),
                        dpath.joinpath(file + ".xyz"),
                    )
                    items.append(file)
            else:
                for file in tqdm(random.sample(names, k=n_items)):
                    shutil.copy(
                        xyz_path.joinpath(file + ".xyz"),
                        dpath.joinpath(file + ".xyz"),
                    )
                    items.append(file)

            df = df[df["file_name"].isin(items)]

            df = df[targets]

            df.to_csv(dpath.joinpath("dataset.csv"))

        else:
            df = Utils.drop_by_oxygen_distribution(
                csv_path=xyz_path.joinpath("dataset.csv"),
                xyz_path=xyz_path,
                threshold=oxygen_distribution_threshold,
                targets=targets,
            )

            if drop_custom:
                df = Utils.drop_custom(df)

            if min_num_atoms is not None:
                df = Utils.drop_min_num_atoms(df, min_num_atoms)

            df.to_csv(dpath.joinpath("dataset.csv"))

            names = df["file_name"].tolist()

            items = []

            for file in tqdm(random.sample(names, k=n_items)):
                shutil.copy(
                    xyz_path.joinpath(file + ".xyz"),
                    dpath.joinpath(file + ".xyz"),
                )
                items.append(file)

            df = df[df["file_name"].isin(items)]

            df = df[targets]

            df.to_csv(dpath.joinpath("dataset.csv"))

        Utils.dataset_max_and_min(spath=dpath, dpath=dpath)

        print("Done")

    @staticmethod
    def drop_min_num_atoms(df: pd.DataFrame, min_num_atoms: int | list):
        print("Dropping outliers from the dataset...\n")

        if isinstance(min_num_atoms, int):
            indices_to_drop = df.index[df["atom_number_total"] < min_num_atoms].tolist()
        elif len(min_num_atoms) == 2:
            indices_down = df.index[df["atom_number_total"] < min_num_atoms[0]].tolist()
            indices_up = df.index[df["atom_number_total"] > min_num_atoms[1]].tolist()
            indices_to_drop = [*indices_down, *indices_up]
        else:
            raise Exception(f"Wrong type for 'min_num_atoms': {type(min_num_atoms)}")

        df = df.drop(indices_to_drop, axis=0)

        return df

    @staticmethod
    def update_yaml(spath: Path, new_value: float | str, target_key="base_lr"):
        def search_and_modify(data, target_key, new_value):
            for key, value in data.items():
                if key == target_key:
                    data[key] = new_value
                    return True
                elif type(value) is dict:
                    if search_and_modify(value, target_key, new_value):
                        return True
            return False

        # Open the YAML file
        with open(str(spath), "r") as file:
            # Load the YAML data into a Python dictionary
            data = yaml.load(file, Loader=yaml.FullLoader)

        if search_and_modify(data, target_key, new_value):
            with open(str(spath), "w") as file:
                yaml.dump(data, file)
            print(f"{target_key} value changed to {new_value}")
        else:
            print(f"{target_key} not found in the file")


class CoulombUtils:
    @staticmethod
    def compute_coulomb_matrix(
        xyz_file: Path, dpath: Path, format: str = ".npy"
    ) -> np.array:
        """
        Compute the Coulomb matrix of a molecule from the 3D coordinates of its atoms as proposed by Rupp et al. in
        "Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning" (2012).

        Parameters:
        xyz_file (Path): The path to the xyz file containing the atomic coordinates.
        dpath (Path): The output path for the txt file.

        Returns:
        np.array : The Coulomb matrix of the molecule

        """
        # Read the xyz file and extract the atomic coordinates
        with open(str(xyz_file), "r") as f:
            lines = f.readlines()
            num_atoms = int(lines[0])
            coordinates = []
            periodic_table = {
                "H": 1,
                "C": 6,
                "O": 8,
            }
            for line in lines[2:]:
                tokens = line.split()
                atom_type = tokens[0]
                nuclear_charge = periodic_table[atom_type]
                x, y, z = map(float, tokens[1:4])
                coordinates.append([nuclear_charge, x, y, z])

        # Compute the Coulomb matrix
        coordinates = np.array(coordinates)
        coulomb_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if i == j:
                    coulomb_matrix[i, j] = 0.5 * coordinates[i, 0] ** 2.4
                else:
                    distance = np.linalg.norm(coordinates[i, 1:] - coordinates[j, 1:])
                    coulomb_matrix[i, j] = (
                        coordinates[i, 0] * coordinates[j, 0] / distance
                    )
                    coulomb_matrix[j, i] = coulomb_matrix[i, j]
        eigenvalues = np.sort(np.linalg.eigvalsh(coulomb_matrix))[::-1]
        coulomb_matrix_sorted = eigenvalues - eigenvalues[:, None]

        # Save the upper triangular matrix to a file
        if format == ".npy":
            np.save(
                dpath.joinpath(xyz_file.stem + format),
                coulomb_matrix_sorted,
            )
        elif format == ".txt":
            np.savetxt(
                dpath.joinpath(xyz_file.stem + format),
                coulomb_matrix_sorted,
                delimiter=",",
            )
        else:
            raise Exception("Unknown format\n")

        return coulomb_matrix_sorted

    @staticmethod
    def fast_compute_coulomb_matrix(
        xyz_file: Path, dpath: Path, format: str = ".npy"
    ) -> np.array:
        """
        Compute the Coulomb matrix of a molecule from the 3D coordinates of its atoms, permute the Coulomb matrix in such a way that the rows (and columns) Ci of the Coulomb matrix are ordered by their norm.

        Parameters:
        xyz_file (Path): The path to the xyz file containing the atomic coordinates.
        dpath (Path): The output path for the txt file.

        Returns:
        np.ndarray : The Coulomb matrix of the molecule

        """
        # Read the xyz file and extract the atomic coordinates
        with open(str(xyz_file), "r") as f:
            lines = f.readlines()
            num_atoms = int(lines[0])
            coordinates = []
            periodic_table = {
                "H": 1,
                "C": 6,
                "O": 8,
            }
            for line in lines[2:]:
                tokens = line.split()
                atom_type = tokens[0]
                nuclear_charge = periodic_table[atom_type]
                x, y, z = map(float, tokens[1:4])
                coordinates.append([nuclear_charge, x, y, z])

        # Compute the Coulomb matrix
        coordinates = np.array(coordinates)
        coulomb_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if i == j:
                    coulomb_matrix[i, j] = 0.5 * coordinates[i, 0] ** 2.4
                else:
                    distance = np.linalg.norm(coordinates[i, 1:] - coordinates[j, 1:])
                    coulomb_matrix[i, j] = (
                        coordinates[i, 0] * coordinates[j, 0] / distance
                    )
                    coulomb_matrix[j, i] = coulomb_matrix[i, j]
        norm = np.linalg.norm(coulomb_matrix, axis=1)
        sort_indices = np.argsort(norm)[::-1]
        sorted_matrix = coulomb_matrix[sort_indices]
        sorted_matrix = sorted_matrix[:, sort_indices]

        # Save the upper triangular matrix to a file
        if format == ".npy":
            np.save(
                dpath.joinpath(xyz_file.stem + format),
                sorted_matrix,
            )
        elif format == ".txt":
            np.savetxt(
                dpath.joinpath(xyz_file.stem + format),
                sorted_matrix,
                delimiter=",",
            )
        else:
            raise Exception("Unknown format\n")

        return sorted_matrix

    @staticmethod
    def generate_coulomb_matrices(spath: Path, dpath: Path, fast: False):
        dpath.mkdir(parents=True, exist_ok=True)

        items = [f for f in spath.iterdir() if f.suffix == ".xyz"]
        pbar = tqdm(total=len(items))

        for i in items:
            (
                CoulombUtils.fast_compute_coulomb_matrix(i, dpath)
                if fast
                else CoulombUtils.compute_coulomb_matrix(i, dpath)
            )
            pbar.update(1)
        pbar.close()

        print("Done")

    @staticmethod
    def restore_symmetric_matrix(matrix: np.array):
        # Reconstruct the symmetric matrix
        symmetric_matrix = matrix - np.transpose(matrix)

        return symmetric_matrix

    @staticmethod
    def find_maximum_shape(folder: Path, format: str = ".npy"):
        max_shape = (0, 0)
        for file in tqdm(folder.glob(f"*{format}")):
            matrix = np.load(file)
            shape = matrix.shape
            if shape[0] > max_shape[0] and shape[1] > max_shape[1]:
                max_shape = shape
        np.savetxt(str(folder.joinpath("max_shape.txt")), max_shape)
        return max_shape


class OxygenUtils:
    @staticmethod
    def compute_mean_oxygen_distance(file_path: Path):
        # Load the file into numpy arrays
        X, Y, Z, atoms = Utils.read_from_xyz_file(file_path)

        # Identify oxygen atoms and extract their positions
        oxygen_indices = [i for i, atom in enumerate(atoms) if atom == "O"]
        oxygen_positions = np.column_stack(
            (X[oxygen_indices], Y[oxygen_indices], Z[oxygen_indices])
        )

        # Compute pairwise distances between all oxygen atoms
        distances = pdist(oxygen_positions)

        # Compute the mean distance between all oxygen atoms
        mean_oxygen_distance = np.mean(distances)

        return mean_oxygen_distance

    @staticmethod
    def get_oxygen_distribution_means(
        csv_path: Path,
        xyz_path: Path,
        dpath: Path = None,
        targets: list = [
            "electronegativity",
            "total_energy",
            "electron_affinity",
            "ionization_potential",
            "Fermi_energy",
        ],
    ) -> np.array:
        MAX, MIN = OxygenUtils.find_max_min_distribution(csv_path, xyz_path)

        df = pd.read_csv(str(csv_path))
        df = df.dropna(subset=targets)

        indices = []
        for t in targets:
            idx = df.index[df[t] == 0.0].tolist()
            indices = [*indices, *idx]
        df = df.drop(indices, axis=0)

        names = df["file_name"].tolist()

        distribution_means = []

        for name in tqdm(names):
            distribution_means.append(
                OxygenUtils.compute_mean_oxygen_distance(
                    xyz_path.joinpath(name + ".xyz")
                )
            )

        df["distribution_mean"] = distribution_means
        df.to_csv(csv_path.with_stem("distribution_mean"))

    @staticmethod
    def compute_correlation(
        csv_path: Path, distribution_path: Path, dpath: Path, normalize: bool = True
    ):
        dpath.mkdir(parents=True, exist_ok=True)
        # Load data
        df = pd.read_csv(str(csv_path))
        distribution = np.loadtxt(str(distribution_path))

        df["electronegativity"].fillna(0, inplace=True)
        print(len(df))

        if normalize:
            var = np.array(df["total_energy"])
            max = np.max(var)
            min = np.min(var)
            var = (var - min) / (max - min) + 0.0001
            print(var)
            df["total_energy"] = var

            var = np.array(df["electronegativity"])
            max = np.max(var)
            min = np.min(var)
            var = (var - min) / (max - min) + 0.0001
            print(var)
            df["electronegativity"] = var

            max_distribution = np.max(distribution)
            min_distribution = np.min(distribution)
            distribution = (distribution - min_distribution) / (
                max_distribution - min_distribution
            ) + 0.0001

        # Scatter plot
        plt.scatter(np.exp(df["electronegativity"]), np.exp(distribution))
        plt.xlabel("electronegativity")
        plt.ylabel("distribution")
        plt.title("Scatter plot")
        plt.savefig(str(dpath.joinpath("scatter_exp.png")))
        plt.close()

        # Line plot
        plt.plot(np.exp(df["electronegativity"]), np.exp(distribution))
        plt.xlabel("electronegativity")
        plt.ylabel("distribution")
        plt.title("Line plot")
        plt.savefig(str(dpath.joinpath("line_exp.png")))
        plt.close()

        plt.hist(
            distribution,
            bins=30,
        )

        # Add labels and title
        plt.xlabel("Mean Distribution")
        plt.ylabel("Frequency")
        plt.title("Histogram of Data")

        # Show the plot
        plt.savefig(str(dpath.joinpath("distribution_means.png")))
        plt.close()

    @staticmethod
    def copy_distributions(dataset_path: Path, distribution_path: Path):
        train_path = dataset_path.joinpath("train")
        val_path = dataset_path.joinpath("val")
        test_path = dataset_path.joinpath("test")

        train_items = [item for item in train_path.iterdir() if item.suffix == ".png"]
        val_items = [item for item in val_path.iterdir() if item.suffix == ".png"]
        test_items = [item for item in test_path.iterdir() if item.suffix == ".png"]

        for item in tqdm(train_items):
            shutil.copy(
                str(distribution_path.joinpath(item.stem + "_distribution.txt")),
                str(train_path.joinpath(item.stem + "_distribution.txt")),
            )

        for item in tqdm(val_items):
            shutil.copy(
                str(distribution_path.joinpath(item.stem + "_distribution.txt")),
                str(val_path.joinpath(item.stem + "_distribution.txt")),
            )

        for item in tqdm(test_items):
            shutil.copy(
                str(distribution_path.joinpath(item.stem + "_distribution.txt")),
                str(test_path.joinpath(item.stem + "_distribution.txt")),
            )


if __name__ == "__main__":
    pass
