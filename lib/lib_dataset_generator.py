try:

    from tqdm import tqdm
    from lib.lib_utils import Utils
    import pandas as pd
    import shutil
    from pathlib import Path

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

        if not self.csv_flag:
            DatasetGenerator.generate_cropped_png_dataset_from_xyz(
                spath=self.path_xyz, dpath=self.spath
            )
            self.split_dataset()
            Utils.generate_num_atoms(dataset_path=self.dpath, xyz_path=self.path_xyz)
            Utils.find_max_dimensions_png_folder(spath=self.spath, dpath=self.dpath)
        else:
            self.split_dataset_from_csv()

    def split_dataset(self):
        # split the dataset in train and test set
        print("Splitting the dataset in train/test/validation set...\n")
        Utils.train_val_test_split_png(
            self.spath,
            self.dpath,
            self.path_csv,
            features=["file_name", *self.features],
            split=self.train_split,
            val_split=self.val_split,
            shuffle=self.shuffle,
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
