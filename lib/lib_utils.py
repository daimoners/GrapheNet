try:

    from datetime import datetime
    import seaborn as sns
    import numpy as np
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
    import cupy as cp
    import math
    from scipy.stats import pearsonr, spearmanr, kendalltau
    import yaml
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

except Exception as e:

    print("Some module are missing {}".format(e))


class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

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
                if len(l) == 4:
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
        resolution=320,
        z_relative=False,
        single_channel_images=False,
    ):
        """Generate a .npy matrix starting from lists of x,y,z coordinates"""

        X, Y, Z, atoms = Utils.read_from_xyz_file(spath)

        C = np.zeros((resolution, resolution))
        O = np.zeros((resolution, resolution))
        H = np.zeros((resolution, resolution))

        if z_relative:
            z_max = np.max(Z)
            z_min = np.min(Z)
        else:
            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            z_max = x[0][2]
            z_min = x[1][2]

        z_norm = lambda x: (x - z_min) / (z_max - z_min)

        C_only = True

        for i in range(len(X)):
            if atoms[i] == "C":
                x_coord = int(round(X[i] * resolution / 160) + resolution / 2)
                y_coord = int(round(Y[i] * resolution / 160) + resolution / 2)
                if C[y_coord, x_coord] < z_norm(Z[i]):
                    C[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "O":
                C_only = False
                x_coord = int(round(X[i] * resolution / 160) + resolution / 2)
                y_coord = int(round(Y[i] * resolution / 160) + resolution / 2)
                if O[y_coord, x_coord] < z_norm(Z[i]):
                    O[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "H":
                C_only = False
                x_coord = int(round(X[i] * resolution / 160) + resolution / 2)
                y_coord = int(round(Y[i] * resolution / 160) + resolution / 2)
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

    @staticmethod
    def dataset_max_and_min(spath: Path, dpath: Path = None) -> list:
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
                if len(l) == 4:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        max = [np.max(X), np.max(Y), np.max(Z)]
        min = [np.min(X), np.min(Y), np.min(Z)]

        return max, min

    # @staticmethod
    # def train_val_test_split_png( #!DEPRECATED
    #     spath: Path,
    #     dpath: Path,
    #     csv: Path,
    #     features: list,
    #     split: float = 0.7,
    #     val_split: float = 0.15,
    #     shuffle: bool = False,
    # ):
    #     """Split a dataset in test and train and generate the respective CSV files"""

    #     train_path = dpath.joinpath("train")
    #     test_path = dpath.joinpath("test")
    #     val_path = dpath.joinpath("val")

    #     train_path.mkdir(parents=True, exist_ok=True)
    #     test_path.mkdir(parents=True, exist_ok=True)
    #     val_path.mkdir(parents=True, exist_ok=True)

    #     data = pd.read_csv(csv)
    #     n_items = len(data.index)

    #     n_train = round(split * n_items)
    #     n_val = round(val_split * n_items)
    #     n_test = n_items - n_train - n_val

    #     print(
    #         f"Train items = {n_train}, Validation items = {n_val}, Test items = {n_test}\n"
    #     )

    #     if shuffle:
    #         indices = random.sample(range(n_items), n_items)

    #     # Start moving train sample

    #     pbar = tqdm(total=n_train)
    #     for i in range(n_train):
    #         file = (
    #             data["file_name"][i] + ".png"
    #             if not shuffle
    #             else data["file_name"][indices[i]] + ".png"
    #         )
    #         shutil.copy(spath.joinpath(file), train_path.joinpath(file))
    #         pbar.update(1)
    #     pbar.close()

    #     train_csv = (
    #         data.loc[0:(n_train), features]
    #         if not shuffle
    #         else data.loc[data.index[indices[0:(n_train)]], features]
    #     )
    #     train_csv.to_csv(train_path.joinpath("train.csv"))

    #     print("Train files moved\n")

    #     # Start moving validation sample

    #     pbar = tqdm(total=n_val)
    #     for i in range(n_val):
    #         file = (
    #             data["file_name"][i + n_train] + ".png"
    #             if not shuffle
    #             else data["file_name"][indices[i + n_train]] + ".png"
    #         )
    #         shutil.copy(spath.joinpath(file), val_path.joinpath(file))
    #         pbar.update(1)
    #     pbar.close()

    #     val_csv = (
    #         data.loc[(n_train) : (n_train + n_val), features]
    #         if not shuffle
    #         else data.loc[data.index[indices[(n_train) : (n_train + n_val)]], features]
    #     )
    #     val_csv.to_csv(val_path.joinpath("val.csv"))

    #     print("Validation files moved\n")

    #     # Start moving test sample

    #     pbar = tqdm(total=n_test)
    #     for i in range(n_test):
    #         file = (
    #             data["file_name"][i + n_train + n_val] + ".png"
    #             if not shuffle
    #             else data["file_name"][indices[i + n_train + n_val]] + ".png"
    #         )
    #         shutil.copy(spath.joinpath(file), test_path.joinpath(file))
    #         pbar.update(1)
    #     pbar.close()

    #     test_csv = (
    #         data.loc[(n_train + n_val) : (n_items + 1), features]
    #         if not shuffle
    #         else data.loc[
    #             data.index[indices[(n_train + n_val) : (n_items + 1)]], features
    #         ]
    #     )
    #     test_csv.to_csv(test_path.joinpath("test.csv"))

    #     shutil.copy(
    #         csv,
    #         dpath.joinpath("dataset.csv"),
    #     )

    #     # try:
    #     #     shutil.copy(
    #     #         csv.parent.joinpath("max_min_coordinates.txt"), #! non mi serve questa parte
    #     #         dpath.joinpath("max_min_coordinates.txt"),
    #     #     )
    #     # except:
    #     #     pass

    #     print("Test files moved\n")

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
            shutil.copy(
                spath.joinpath(name + suffix), train_path.joinpath(name + suffix)
            )
        train_csv.to_csv(train_path.joinpath("train.csv"))

        print("Train files moved\n")

        # Start moving validation sample

        val_csv = df.loc[val_indices, features]
        names = val_csv["file_name"].tolist()
        for name in tqdm(names):
            shutil.copy(spath.joinpath(name + suffix), val_path.joinpath(name + suffix))
        val_csv.to_csv(val_path.joinpath("val.csv"))

        print("Validation files moved\n")

        # Start moving test sample

        test_csv = df.loc[test_indices, features]
        names = test_csv["file_name"].tolist()
        for name in tqdm(names):
            shutil.copy(
                spath.joinpath(name + suffix), test_path.joinpath(name + suffix)
            )
        test_csv.to_csv(test_path.joinpath("test.csv"))

        print("Test files moved\n")

        df = df.loc[:, features]
        df.to_csv(dpath.joinpath("dataset.csv"))

    @staticmethod
    def stratified_train_val_test_split_png(
        spath: Path,
        dpath: Path,
        csv: Path,
        features: list,
        target_focus: str,
        split: float = 0.7,
        val_split: float = 0.15,
    ):
        """Split a dataset in test and train and generate the respective CSV files"""

        print("Stratified split...")

        train_path = dpath.joinpath("train")
        test_path = dpath.joinpath("test")
        val_path = dpath.joinpath("val")

        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        train_csv, val_csv, test_csv = Utils.stratified_split(
            csv_path=csv, target=target_focus, split=[split, val_split, val_split]
        )

        # Start moving train sample
        names = train_csv["file_name"].tolist()

        for name in tqdm(names):
            shutil.copy(
                spath.joinpath(name + ".png"), train_path.joinpath(name + ".png")
            )
        train_csv = train_csv.loc[:, features]
        train_csv.to_csv(train_path.joinpath("train.csv"))
        print("Train files moved\n")

        # Start moving validation sample
        names = val_csv["file_name"].tolist()

        for name in tqdm(names):
            shutil.copy(spath.joinpath(name + ".png"), val_path.joinpath(name + ".png"))
        val_csv = val_csv.loc[:, features]
        val_csv.to_csv(val_path.joinpath("val.csv"))
        print("Validation files moved\n")

        # Start moving test sample
        names = test_csv["file_name"].tolist()

        for name in tqdm(names):
            shutil.copy(
                spath.joinpath(name + ".png"), test_path.joinpath(name + ".png")
            )
        test_csv = test_csv.loc[:, features]
        test_csv.to_csv(test_path.joinpath("test.csv"))
        print("Test files moved\n")

        shutil.copy(
            csv,
            dpath.joinpath("dataset.csv"),
        )

        # try:
        #     shutil.copy(
        #         csv.parent.joinpath("max_min_coordinates.txt"), #! non mi serve questa parte
        #         dpath.joinpath("max_min_coordinates.txt"),
        #     )
        # except:
        #     pass

        print("Test files moved\n")

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
    def generate_num_atoms(dataset_path: Path, xyz_path: Path, format=".png"):
        """Generate the number and type of atoms in a .txt file for each .xyz file

        Args:
            dataset_path (Path): path of the dataset already splitted in train/val/test
            xyz_path (Path): path of all the .xyz files
            format (str, optional): image format. Defaults to ".png".
        """

        for dir in ["train", "val", "test"]:
            for file in tqdm(dataset_path.joinpath(dir).iterdir()):

                if file.suffix == format:

                    X, Y, Z, atoms = Utils.read_from_xyz_file(
                        xyz_path.joinpath(file.stem + ".xyz")
                    )

                    n_C = atoms.count("C")
                    n_O = atoms.count("O")
                    n_H = atoms.count("H")

                    lines = []
                    with open(dataset_path.joinpath(dir, file.stem + ".txt"), "w") as f:

                        lines.append(f"{n_C}\n") if n_C > 0 else None
                        lines.append(f"{n_O}\n") if n_O > 0 else None
                        lines.append(f"{n_H}\n") if n_H > 0 else None

                        f.writelines(lines)

    @staticmethod
    def plot_fit(y: list, y_hat: list, dpath: Path, target: str):
        def r2_score(y_pred, y_true):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            mean_y = np.mean(y_true)
            SSR = np.sum((y_pred - y_true) ** 2)
            SST = np.sum((y_true - mean_y) ** 2)
            return 1 - SSR / SST

        min = np.min([np.min(y), np.min(y_hat)])
        max = np.max([np.max(y), np.max(y_hat)])

        plt.figure(figsize=(10, 7))
        plt.plot(
            [min, max],
            [min, max],
        )
        plt.scatter(y_hat, y, color="red")
        plt.xlabel("Predictions")
        plt.ylabel("Targets")
        plt.title(f"{target} - R2 = {r2_score(y_hat,y):.3f}")
        plt.savefig(str(dpath))

    @staticmethod
    def drop_custom(
        df: pd.DataFrame,
    ):

        print("Dropping outliers from the dataset...\n")

        el_aff_down = df.index[df["electron_affinity"] <= -7].tolist()
        el_aff_up = df.index[df["electron_affinity"] >= -3.5].tolist()
        el_aff = [*el_aff_down, *el_aff_up]

        elneg_down = df.index[df["electronegativity"] <= -6.5].tolist()
        elneg_up = df.index[df["electronegativity"] >= -3.5].tolist()
        elneg = [*elneg_down, *elneg_up]

        i_pot_down = df.index[df["ionization_potential"] <= -6.2].tolist()
        i_pot_up = df.index[df["ionization_potential"] >= -3.5].tolist()
        i_pot = [*i_pot_down, *i_pot_up]

        indices = [*el_aff, *elneg, *i_pot]

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
        df = df.drop(indices, axis=0)

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
    ):

        targets = ["file_name", *targets]

        dpath.mkdir(parents=True, exist_ok=True)

        if (
            oxygen_distribution_threshold is None
            or oxygen_distribution_threshold == 0.0
        ):

            samples = [x for x in xyz_path.iterdir() if x.suffix == ".xyz"]

            df = pd.read_csv(xyz_path.joinpath("dataset.csv"))

            df = Utils.drop_nan_and_zeros(df)

            df = Utils.drop_custom(df)

            items = []

            for file in tqdm(random.sample(samples, k=n_items)):
                shutil.copy(
                    file,
                    dpath.joinpath(file.name),
                )
                items.append(file.stem)

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

    @staticmethod
    def stratified_split(
        csv_path: Path, target: str, split: list = [0.7, 0.15, 0.15]
    ):  #! questo serve per fare meglio lo split
        def assign_classes(array, start, stop, step=1):
            # Define the range for each class

            class_ranges = [
                (start + i * step, start + (i + 1) * step)
                for i in range(int((stop - start) / step))
            ]

            # Create an empty list to store the classes
            classes = []

            # Loop through each float value in the array
            for value in array:
                # Loop through each class range
                for i, (low, high) in enumerate(class_ranges):
                    # If the value falls within the class range, add the class label to the list
                    if low < value <= high:
                        classes.append(i)
                        break

            # Return the list of classes as a numpy array
            return np.array(classes)

        df = pd.read_csv(csv_path)

        df = Utils.drop_nan_and_zeros(df)

        X = np.zeros(len(df))

        y = df[target].values
        print(len(y))
        max = np.max(y) + 0.5
        min = np.min(y) - 0.5
        y = assign_classes(y, start=min, stop=max, step=1)

        # Split the data into training, validation, and test sets for target 1
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

        for train_index, test_valid_index in sss.split(X, y):
            df_train, df_test_valid = (
                df.iloc[train_index],
                df.iloc[test_valid_index],
            )
            X_train, X_test_valid = X[train_index], X[test_valid_index]
            y_train, y_test_valid = y[train_index], y[test_valid_index]

        # Split the data into training, validation, and test sets for target 1
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

        for valid_index, test_index in sss.split(X_test_valid, y_test_valid):
            df_val, df_test = (
                df_test_valid.iloc[valid_index],
                df_test_valid.iloc[test_index],
            )

        # df_test.to_csv(csv_path.with_name("test.csv"))
        # df_val.to_csv(csv_path.with_name("val.csv"))
        # df_train.to_csv(csv_path.with_name("train.csv"))

        # sns.set_style("white")

        # for i in ["train", "test", "val"]:

        #     if i == "train":
        #         x = df_train[target]
        #     elif i == "test":
        #         x = df_test[target]
        #     elif i == "val":
        #         x = df_val[target]

        #     # Plot
        #     kwargs = dict(hist_kws={"alpha": 0.6}, kde_kws={"linewidth": 2})

        #     plt.figure(figsize=(15, 10), dpi=100)
        #     try:
        #         sns.distplot(x, color="purple", label=target, **kwargs)
        #     except:
        #         pass
        #     plt.legend()
        #     plt.savefig(csv_path.parent.joinpath(f"{i}.png"))

        return df_train, df_val, df_test


class CoulombUtils:
    @staticmethod
    def compute_coulomb_matrix(xyz_file: Path, dpath: Path) -> cp.array:
        """
        Compute the Coulomb matrix of a molecule from the 3D coordinates of its atoms as proposed by Rupp et al. in
        "Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning" (2012).

        Parameters:
        xyz_file (Path): The path to the xyz file containing the atomic coordinates.
        dpath (Path): The output path for the txt file.

        Returns:
        cp.array : The Coulomb matrix of the molecule

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
        coordinates = cp.array(coordinates)
        coulomb_matrix = cp.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if i == j:
                    coulomb_matrix[i, j] = 0.5 * coordinates[i, 0] ** 2.4
                else:
                    distance = cp.linalg.norm(coordinates[i, 1:] - coordinates[j, 1:])
                    coulomb_matrix[i, j] = (
                        coordinates[i, 0] * coordinates[j, 0] / distance
                    )
                    coulomb_matrix[j, i] = coulomb_matrix[i, j]
        eigenvalues = cp.sort(cp.linalg.eigvalsh(coulomb_matrix))[::-1]
        coulomb_matrix_sorted = eigenvalues - eigenvalues[:, None]

        upper_triangular = cp.triu(coulomb_matrix_sorted)

        # Convert the cupy array to numpy array
        upper_triangular_np = cp.asnumpy(upper_triangular)

        # Save the upper triangular matrix to a file
        np.savetxt(
            dpath.joinpath(xyz_file.stem + ".txt"),
            upper_triangular_np,
            delimiter=",",
        )

        return coulomb_matrix_sorted

    @staticmethod
    def fast_compute_coulomb_matrix(xyz_file: Path, dpath: Path) -> cp.array:
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
        coordinates = cp.array(coordinates)
        coulomb_matrix = cp.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if i == j:
                    coulomb_matrix[i, j] = 0.5 * coordinates[i, 0] ** 2.4
                else:
                    distance = cp.linalg.norm(coordinates[i, 1:] - coordinates[j, 1:])
                    coulomb_matrix[i, j] = (
                        coordinates[i, 0] * coordinates[j, 0] / distance
                    )
                    coulomb_matrix[j, i] = coulomb_matrix[i, j]
        norm = cp.linalg.norm(coulomb_matrix, axis=1)
        sort_indices = cp.argsort(norm)[::-1]
        sorted_matrix = coulomb_matrix[sort_indices]
        sorted_matrix = sorted_matrix[:, sort_indices]

        upper_triangular = cp.triu(sorted_matrix)

        # Convert the cupy array to numpy array
        upper_triangular_np = cp.asnumpy(upper_triangular)

        # Save the upper triangular matrix to a file
        np.savetxt(
            dpath.joinpath(xyz_file.stem + ".txt"),
            upper_triangular_np,
            delimiter=",",
        )

        return sorted_matrix

    @staticmethod
    def generate_coulomb_matrices(spath: Path, dpath: Path, fast: False):

        dpath.mkdir(parents=True, exist_ok=True)

        items = [f for f in spath.iterdir() if f.suffix == ".xyz"]
        pbar = tqdm(total=len(items))

        for i in items:
            CoulombUtils.fast_compute_coulomb_matrix(
                i, dpath
            ) if fast else CoulombUtils.compute_coulomb_matrix(i, dpath)
            pbar.update(1)
        pbar.close()

        print("Done")

    @staticmethod
    def restore_symmetric_matrix(matrix: np.array):

        # Reconstruct the symmetric matrix
        symmetric_matrix = matrix - cp.transpose(matrix)

        return symmetric_matrix


class OxygenUtils:
    @staticmethod
    def get_oxygen_distribution_means(
        csv_path: Path, xyz_path: Path, dpath: Path = None
    ) -> np.array:

        MAX, MIN = OxygenUtils.find_max_min_distribution(csv_path, xyz_path)

        df = pd.read_csv(str(csv_path))
        names = df["file_name"].tolist()

        distribution_means = []
        distribution = []
        d = []

        for f in tqdm(range(len(names))):

            X, Y, Z, atoms = Utils.read_from_xyz_file(
                xyz_path.joinpath(names[f] + ".xyz")
            )

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

            distribution_means.append(np.mean(distribution))

            if dpath is not None:
                plt.hist(
                    distribution,
                    bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )

                # Add labels and title
                plt.xlabel("Data")
                plt.ylabel("Frequency")
                plt.title("Histogram of Data")

                # Show the plot
                plt.savefig(dpath.joinpath("distributions", names[f] + ".png"))
                plt.close()

                hist_data = np.histogram(
                    distribution,
                    bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
                np.savetxt(
                    str(
                        dpath.joinpath("distributions", f"{names[f]}_distribution.txt")
                    ),
                    hist_data[
                        0
                    ],  # TODO va normalizzata prima di salvarla, capire che normalizzazione devo fare
                )

        if dpath is not None:
            np.savetxt(
                str(dpath.joinpath("distribution_means.txt")),
                np.array(distribution_means),
            )

        return distribution_means

    @staticmethod
    def find_max_min_distribution(
        csv_path: Path,
        xyz_path: Path,
        targets: list = [
            "electronegativity",
            "total_energy",
            "electron_affinity",
            "ionization_potential",
            "Fermi_energy",
        ],
    ):
        print("Finding MAX and MIN value of the oxygen distribution...\n")
        df = pd.read_csv(str(csv_path))
        df = df.dropna(subset=targets)

        indices = []
        for t in targets:
            idx = df.index[df[t] == 0.0].tolist()
            indices = [*indices, *idx]
        df = df.drop(indices, axis=0)

        names = df["file_name"].tolist()

        max_distribution = []
        min_distribution = []

        distribution = []
        d = []

        for f in range(len(names)):

            X, Y, Z, atoms = Utils.read_from_xyz_file(
                xyz_path.joinpath(names[f] + ".xyz")
            )

            distribution.clear()

            for i in range(len(X)):
                if atoms[i] == "O":
                    d.clear()
                    for j in range(len(X)):
                        if atoms[j] == "O" and i != j:
                            P1 = [X[i], Y[i]]
                            P2 = [X[j], Y[j]]
                            d.append(math.dist(P1, P2))

                    distribution.append(np.mean(d))

            max_distribution.append(np.max(distribution))
            min_distribution.append(np.min(distribution))

        return np.max(max_distribution), np.min(min_distribution)

    @staticmethod
    def compute_correlation(
        csv_path: Path, distribution_path: Path, dpath: Path, normalize: bool = True
    ):
        dpath.mkdir(parents=True, exist_ok=True)
        # Load data
        df = pd.read_csv(str(csv_path))
        distribution = np.loadtxt(str(distribution_path))

        # df = df.dropna(
        #     subset=[
        #         "electronegativity",
        #         "total_energy",
        #         "electron_affinity",
        #         "ionization_potential",
        #         "Fermi_energy",
        #     ]
        # )
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

        # # Calculate Pearson correlation coefficient
        # pearson_corr, p_value = pearsonr(df["total_energy"], distribution)
        # print("Pearson correlation coefficient:", pearson_corr)

        # # Calculate Spearman rank correlation coefficient
        # spear_corr, p_value = spearmanr(df["total_energy"], distribution)
        # print("Spearman rank correlation coefficient:", spear_corr)

        # # Calculate Kendall rank correlation coefficient
        # kendall_corr, p_value = kendalltau(df["total_energy"], distribution)
        # print("Kendall rank correlation coefficient:", kendall_corr)

        # correlation = {
        #     "Pearson correlation coefficient": float(pearson_corr),
        #     "Spearman rank correlation coefficient": float(spear_corr),
        #     "Kendall rank correlation coefficient": float(kendall_corr),
        # }
        # with open(
        #     str(dpath.joinpath("correlation.yaml")),
        #     "w",
        # ) as outfile:
        #     yaml.dump(correlation, outfile)

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
    # Utils.drop_outliers(
    #     spath=Path("/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt"),
    #     dpath=Path(__file__).parent.parent.joinpath("data_GO", "filtered_xyz"),
    # )
    # Utils.create_subset_xyz(
    #     xyz_path=Path(__file__).parent.parent.joinpath("data_GO", "filtered_xyz"),
    #     dpath=Path(__file__).parent.parent.joinpath("data_GO", "subset_xyz"),
    #     n_items=7000,
    #     targets=[
    #         "total_energy",
    #         "ionization_potential",
    #         "electronegativity",
    #         "electron_affinity",
    #         "band_gap",
    #         "Fermi_energy",
    #     ],
    # )
    # Utils.find_max_dimensions_png_folder(
    #     spath=Path(__file__).parent.parent.joinpath("data_GO", "subset_png"),
    #     dpath=Path(__file__).parent.parent.joinpath("data_GO", "training_dataset"),
    # )
    # Utils.drop_custom(
    #     spath=Path("/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt"),
    #     dpath=Path(__file__).parent.parent.joinpath("data_GO", "custom_xyz"),
    # )
    # CoulombUtils.generate_coulomb_matrices(
    #     spath=Path(__file__).parent.parent.joinpath("data_GO", "custom_subset_xyz"),
    #     dpath=Path(__file__).parent.parent.joinpath("data_GO", "custom_coulomb"),
    #     fast=True,
    # )
    # OxygenUtils.get_oxygen_distribution_means(
    #     csv_path=Path(
    #         "/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt/dataset.csv"
    #     ),
    #     xyz_path=Path("/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt"),
    #     dpath=Path(__file__).parent.parent.joinpath("tests", "oxygens"),
    # )
    # OxygenUtils.compute_correlation(
    #     csv_path=Path(
    #         "/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt/dataset.csv"
    #     ),
    #     distribution_path=Path(__file__).parent.parent.joinpath(
    #         "tests", "oxygens", "distribution_means.txt"
    #     ),
    #     dpath=Path(__file__).parent.parent.joinpath("tests", "oxygens", "results"),
    # )
    # OxygenUtils.copy_distributions(
    #     dataset_path=Path(__file__).parent.parent.joinpath(
    #         "data_GO", "custom_training_dataset"
    #     ),
    #     distribution_path=Path(__file__).parent.parent.joinpath(
    #         "tests", "oxygens", "distributions"
    #     ),
    # )
    Utils.stratified_split2(
        csv_path=Path(
            "/home/cnrismn/git_workspace/GrapheNet/data_GO/training_dataset/dataset.csv"
        ),
        target="electron_affinity",
    )
