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

    @staticmethod
    def train_val_test_split_png(
        spath: Path,
        dpath: Path,
        csv: Path,
        features: list,
        split: float = 0.7,
        val_split: float = 0.15,
        shuffle: bool = False,
    ):
        """Split a dataset in test and train and generate the respective CSV files"""

        train_path = dpath.joinpath("train")
        test_path = dpath.joinpath("test")
        val_path = dpath.joinpath("val")

        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(csv)
        n_items = len(data.index)

        n_train = round(split * n_items)
        n_val = round(val_split * n_items)
        n_test = n_items - n_train - n_val

        print(
            f"Train items = {n_train}, Validation items = {n_val}, Test items = {n_test}\n"
        )

        if shuffle:
            indices = random.sample(range(n_items), n_items)

        # Start moving train sample

        pbar = tqdm(total=n_train)
        for i in range(n_train):
            file = (
                data["file_name"][i] + ".png"
                if not shuffle
                else data["file_name"][indices[i]] + ".png"
            )
            shutil.copy(spath.joinpath(file), train_path.joinpath(file))
            pbar.update(1)
        pbar.close()

        train_csv = (
            data.loc[0:(n_train), features]
            if not shuffle
            else data.loc[data.index[indices[0:(n_train)]], features]
        )
        train_csv.to_csv(train_path.joinpath("train.csv"))

        print("Train files moved\n")

        # Start moving validation sample

        pbar = tqdm(total=n_val)
        for i in range(n_val):
            file = (
                data["file_name"][i + n_train] + ".png"
                if not shuffle
                else data["file_name"][indices[i + n_train]] + ".png"
            )
            shutil.copy(spath.joinpath(file), val_path.joinpath(file))
            pbar.update(1)
        pbar.close()

        val_csv = (
            data.loc[(n_train) : (n_train + n_val), features]
            if not shuffle
            else data.loc[data.index[indices[(n_train) : (n_train + n_val)]], features]
        )
        val_csv.to_csv(val_path.joinpath("val.csv"))

        print("Validation files moved\n")

        # Start moving test sample

        pbar = tqdm(total=n_test)
        for i in range(n_test):
            file = (
                data["file_name"][i + n_train + n_val] + ".png"
                if not shuffle
                else data["file_name"][indices[i + n_train + n_val]] + ".png"
            )
            shutil.copy(spath.joinpath(file), test_path.joinpath(file))
            pbar.update(1)
        pbar.close()

        test_csv = (
            data.loc[(n_train + n_val) : (n_items + 1), features]
            if not shuffle
            else data.loc[
                data.index[indices[(n_train + n_val) : (n_items + 1)]], features
            ]
        )
        test_csv.to_csv(test_path.joinpath("test.csv"))

        shutil.copy(
            csv,
            dpath.joinpath("dataset.csv"),
        )

        try:
            shutil.copy(
                csv.parent.joinpath("max_min_coordinates.txt"),
                dpath.joinpath("max_min_coordinates.txt"),
            )
        except:
            pass

        print("Test files moved\n")

    @staticmethod
    def find_max_dimensions_png_folder(spath: Path):
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

        return np.max(widths), np.max(heights)

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
    def create_subset_xyz(xyz_path: Path, dpath: Path, n_items: int):

        dpath.mkdir(parents=True, exist_ok=True)

        samples = [x for x in xyz_path.iterdir() if x.suffix == ".xyz"]

        df = pd.read_csv(xyz_path.joinpath("dataset.csv"))

        for file in random.sample(samples, k=n_items):
            shutil.copy(
                file,
                dpath.joinpath(file.name),
            )
            df.drop(df[df["file_name"] == str(file.stem)].index, inplace=True)

        df.to_csv(dpath.joinpath("dataset.csv"))

        print("Done")

    @staticmethod
    def compute_min_max_target(
        csv_path: Path, target="total_energy", all_positive=False
    ):
        dataset = pd.read_csv(str(csv_path))

        total_energy = (
            np.abs(np.asarray(dataset[target]))
            if all_positive
            else np.asarray(dataset[target])
        )

        np.savetxt(
            str(csv_path.parent.joinpath(f"min_max_{target}.txt")),
            [np.min(total_energy), np.max(total_energy)],
        )

        print(f"Path: {csv_path}")
        print(f"Min: {np.min(total_energy)}")
        print(f"Max: {np.max(total_energy)}")

    @staticmethod
    def compute_mean_std_target(
        csv_path: Path, target="total_energy", all_positive=False
    ):
        dataset = pd.read_csv(str(csv_path))

        total_energy = (
            np.abs(np.asarray(dataset[target]))
            if all_positive
            else np.asarray(dataset[target])
        )

        np.savetxt(
            str(csv_path.parent.joinpath(f"mean_std_{target}.txt")),
            [np.mean(total_energy), np.std(total_energy)],
        )

        print(f"Path: {csv_path}")
        print(f"Mean: {np.mean(total_energy)}")
        print(f"STD: {np.std(total_energy)}")

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


if __name__ == "__main__":
    Utils.create_subset_xyz(
        xyz_path=Path("/home/cnrismn/git_workspace/Graphene/data/dataset_xyz"),
        dpath=Path(__file__).parent.parent.joinpath("data", "subset_xyz"),
        n_items=7000,
    )
