try:
    from tqdm import tqdm
    from lib.lib_utils import Utils
    import pandas as pd
    import shutil
    from pathlib import Path
    import random
    import cv2
    import numpy as np

except Exception as e:
    print("Some module are missing {}".format(e))


class DatasetGenerator(object):
    @staticmethod
    def generate_cropped_png_dataset_from_xyz(
        spath: Path,
        dpath: Path,
        z_relative=False,
    ):
        dpath.mkdir(parents=True, exist_ok=True)

        if not z_relative and not spath.joinpath("max_min_coordinates.txt").is_file():
            Utils.dataset_max_and_min(spath, spath)

        items = len([f for f in spath.iterdir() if f.suffix == ".xyz" and f.is_file()])
        # generate all the .png files starting from the .xyz files
        print("Generating cropped .png files...\n")
        pbar = tqdm(total=items)
        for file in spath.iterdir():
            if file.suffix == ".xyz":
                Utils.generate_png(
                    file,
                    dpath,
                    z_relative=z_relative,
                )
                pbar.update(1)
        pbar.close()

    @staticmethod
    def drop_custom(
        df: pd.DataFrame,
    ):
        # indices = df.index[df["electron_affinity"] < -5.7].tolist()

        indices = df.index[~df["electron_affinity"].between(-5.9, -5.2)].tolist()

        df = df.drop(indices, axis=0)

        return df

    @staticmethod
    def filter_csv(
        csv_path: Path,
        dpath: Path,
        n_items: int,
        oxygen_outliers_th: float = 0.0,
        min_num_atoms: int | list = 0,
        drop_custom: bool = False,
    ):
        dpath.mkdir(exist_ok=True, parents=True)

        df = pd.read_csv(csv_path)

        if oxygen_outliers_th != 0.0:
            df = df[df["distribution_mean"] > oxygen_outliers_th]
            print(f"Lenght after dropping oxygen distribution outliers: {len(df)}")

        if min_num_atoms != 0:
            if isinstance(min_num_atoms, int):
                df = df[df["atom_number_total"] > min_num_atoms]
            elif isinstance(min_num_atoms, list):
                df = df[
                    df["atom_number_total"].between(min_num_atoms[0], min_num_atoms[1])
                ]
            else:
                raise Exception(f"Wrong class for {min_num_atoms}")
            print(f"Lenght after dropping min num atoms outliers: {len(df)}")

        if drop_custom:
            df = DatasetGenerator.drop_custom(df)
            print(f"Lenght after dropping custom outliers: {len(df)}")

        if n_items == 0:
            df.to_csv(dpath.joinpath("dataset.csv"))

        else:
            items = random.sample(df["file_name"].tolist(), k=n_items)
            df = df[df["file_name"].isin(items)]

            df.to_csv(dpath.joinpath("dataset.csv"))

    @staticmethod
    def copy_xyz_files(
        csv_path: Path, spath: Path, dpath: Path, complete_csv_path: Path = None
    ):
        dpath.mkdir(exist_ok=True, parents=True)

        df = pd.read_csv(csv_path)
        names = df["file_name"].to_list()

        for name in tqdm(names):
            shutil.copy(spath.joinpath(f"{name}.xyz"), dpath.joinpath(f"{name}.xyz"))

        if complete_csv_path is not None:
            complete_df = pd.read_csv(complete_csv_path)
            complete_df = complete_df[complete_df["file_name"].isin(names)]
            complete_df.to_csv(dpath.joinpath("dataset.csv"))

    def __init__(self, cfg):
        self.spath = Path(cfg.spath)
        self.dpath = Path(cfg.dpath)
        self.plot_distributions = cfg.plot_distributions
        self.train_split = cfg.randomly.train_split
        self.test_split = cfg.randomly.test_split
        self.val_split = cfg.randomly.val_split
        self.shuffle = cfg.randomly.shuffle

        self.features = cfg.features
        self.path_csv = Path(cfg.path_csv)
        self.path_xyz = Path(cfg.path_xyz)

        self.csv_dataset_path = Path(cfg.from_csv.csv_dataset_path)

        self.package_path = cfg.package_path

        self.stock_dataset_path = Path(cfg.stock_csv_path)
        self.n_items = cfg.randomly.n_items
        self.oxygen_outliers_th = cfg.randomly.oxygen_outliers_th
        self.min_num_atoms = cfg.randomly.min_num_atoms
        self.drop_custom_flag = cfg.randomly.drop_custom

        self.augmented = cfg.augmented

        if not self.csv_dataset_path.is_file():
            DatasetGenerator.filter_csv(
                csv_path=self.stock_dataset_path,
                dpath=self.path_xyz,
                n_items=self.n_items,
                oxygen_outliers_th=self.oxygen_outliers_th,
                min_num_atoms=self.min_num_atoms,
                drop_custom=self.drop_custom_flag,
            )
            DatasetGenerator.copy_xyz_files(
                csv_path=self.path_xyz.joinpath("dataset.csv"),
                spath=self.stock_dataset_path.parent,
                dpath=self.path_xyz,
            )
        else:
            DatasetGenerator.copy_xyz_files(
                csv_path=self.csv_dataset_path,
                spath=self.stock_dataset_path.parent,
                dpath=self.path_xyz,
                complete_csv_path=self.stock_dataset_path,
            )
        DatasetGenerator.generate_cropped_png_dataset_from_xyz(
            spath=self.path_xyz, dpath=self.spath
        ) if not self.spath.is_dir() else None
        self.split_dataset()
        Utils.generate_num_atoms(dataset_path=self.dpath, xyz_path=self.path_xyz)
        Utils.find_max_dimensions_png_folder(spath=self.spath, dpath=self.dpath)
        if self.augmented:
            self.data_augmentation()
        shutil.rmtree(self.spath)
        shutil.rmtree(self.path_xyz)

    def split_dataset(self):
        # split the dataset in train and test set
        print("Splitting the dataset in train/test/validation set...\n")

        Utils.train_val_test_split_png(
            self.spath,
            self.dpath,
            self.path_csv,
            features=["file_name", *self.features],
            split=self.train_split,
        )

        if self.plot_distributions:
            Utils.plot_distribution(
                spath=self.dpath.joinpath("train"),
                csv=self.path_csv,
                features=self.features,
                dpath=self.dpath.joinpath("train", "distributions"),
            )
            Utils.plot_distribution(
                spath=self.dpath.joinpath("test"),
                csv=self.path_csv,
                features=self.features,
                dpath=self.dpath.joinpath("test", "distributions"),
            )
            Utils.plot_distribution(
                spath=self.dpath.joinpath("val"),
                csv=self.path_csv,
                features=self.features,
                dpath=self.dpath.joinpath("val", "distributions"),
            )

    def data_augmentation(self):
        for dir in ["train", "val", "test"]:
            samples = [
                f
                for f in self.dpath.joinpath(dir).iterdir()
                if (f.suffix == ".png" and not "R" in f.stem)
            ]

            for sample in tqdm(samples):
                img = cv2.imread(str(sample), -1)

                for angle in [1, 2, 3]:
                    if angle == 1:
                        rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 2:
                        rotated_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif angle == 3:
                        rotated_image = cv2.rotate(img, cv2.ROTATE_180)

                    cropped_image = np.array(Utils.crop_image(rotated_image))

                    # Save the rotated image
                    cv2.imwrite(
                        str(sample.with_stem(sample.stem + f"_R{angle}")), cropped_image
                    )


if __name__ == "__main__":
    pass
