try:

    from tqdm import tqdm
    from lib.lib_utils import Utils
    import pandas as pd
    import shutil
    from pathlib import Path
    import random

except Exception as e:

    print("Some module are missing {}".format(e))


class DatasetGenerator(object):
    @staticmethod
    def generate_cropped_png_dataset_from_xyz(
        spath: Path,
        dpath: Path,
        resolution=320,
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
                    resolution,
                    z_relative=z_relative,
                )
                pbar.update(1)
        pbar.close()

    @staticmethod
    def drop_custom(
        df: pd.DataFrame,
    ):

        indices = df.index[df["electron_affinity"] < -5.7].tolist()

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
    def copy_xyz_files(csv_path: Path, spath: Path, dpath: Path):

        dpath.mkdir(exist_ok=True, parents=True)

        df = pd.read_csv(csv_path)
        names = df["file_name"].to_list()

        for name in tqdm(names):
            shutil.copy(spath.joinpath(f"{name}.xyz"), dpath.joinpath(f"{name}.xyz"))

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

        self.csv_flag = cfg.from_csv.flag
        self.csv_dataset_path = Path(cfg.from_csv.csv_dataset_path)

        self.package_path = cfg.package_path

        self.target = cfg.target

        self.stock_dataset_path = Path(cfg.stock_csv_path)
        self.n_items = cfg.randomly.n_items
        self.oxygen_outliers_th = cfg.randomly.oxygen_outliers_th
        self.min_num_atoms = cfg.randomly.min_num_atoms
        self.drop_custom_flag = cfg.randomly.drop_custom

        if not self.csv_flag:
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
            DatasetGenerator.generate_cropped_png_dataset_from_xyz(
                spath=self.path_xyz, dpath=self.spath
            ) if not self.spath.is_dir() else None
            self.split_dataset()
            Utils.generate_num_atoms(dataset_path=self.dpath, xyz_path=self.path_xyz)
            Utils.find_max_dimensions_png_folder(spath=self.spath, dpath=self.dpath)
        else:
            self.split_dataset_from_csv()

    def split_dataset(self):
        # split the dataset in train and test set
        print("Splitting the dataset in train/test/validation set...\n")
        if self.target in self.features:
            Utils.stratified_train_val_test_split_png(
                self.spath,
                self.dpath,
                self.path_csv,
                target_focus=self.target,
                features=["file_name", *self.features],
                split=self.train_split,
                val_split=self.val_split,
            )
        else:
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

    def split_dataset_from_csv(self):

        # split the dataset in train and test set
        print("Splitting the dataset in train/test/validation set...\n")

        train_csv = pd.read_csv(self.csv_dataset_path.joinpath("train", "train.csv"))
        train_items = len(train_csv.index)

        pbar = tqdm(total=train_items)
        for i in range(train_items):
            file = train_csv["file_name"][i] + ".png"
            shutil.copy(
                self.spath.joinpath(file),
                self.dpath.joinpath("train", file),
            )
            pbar.update(1)
        pbar.close()

        test_csv = pd.read_csv(self.csv_dataset_path.joinpath("test", "test.csv"))
        test_items = len(test_csv.index)

        pbar = tqdm(total=test_items)
        for i in range(test_items):
            file = test_csv["file_name"][i] + ".png"
            shutil.copy(
                self.spath.joinpath(file),
                self.dpath.joinpath("test", file),
            )
            pbar.update(1)
        pbar.close()

        val_csv = pd.read_csv(self.csv_dataset_path.joinpath("val", "val.csv"))
        val_items = len(val_csv.index)

        pbar = tqdm(total=val_items)
        for i in range(val_items):
            file = val_csv["file_name"][i] + ".png"
            shutil.copy(
                self.spath.joinpath(file),
                self.dpath.joinpath("val", file),
            )
            pbar.update(1)
        pbar.close()

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

        for f in self.csv_dataset_path.iterdir():
            if f.suffix == ".txt" or f.suffix == ".csv":
                shutil.copy(f, self.dpath.joinpath(f.name))

        shutil.copy(
            self.csv_dataset_path.joinpath("train", "train.csv"),
            self.dpath.joinpath("train", "train.csv"),
        )
        shutil.copy(
            self.csv_dataset_path.joinpath("test", "test.csv"),
            self.dpath.joinpath("test", "test.csv"),
        )
        shutil.copy(
            self.csv_dataset_path.joinpath("val", "val.csv"),
            self.dpath.joinpath("val", "val.csv"),
        )


if __name__ == "__main__":
    pass
