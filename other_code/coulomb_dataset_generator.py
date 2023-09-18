try:

    from lib.lib_utils import CoulombUtils, Utils
    from pathlib import Path
    import pandas as pd
    import shutil
    from tqdm import tqdm

except Exception as e:

    print(f"Some moduel are missing: {e}")


def generate_matrices():
    CoulombUtils.generate_coulomb_matrices(
        spath=Path(__file__).parent.joinpath("data", "xyz_files"),
        dpath=Path(__file__).parent.joinpath("data", "matrices_norm_order"),
        fast=True,
    )
    CoulombUtils.generate_coulomb_matrices(
        spath=Path(__file__).parent.joinpath("data", "xyz_files"),
        dpath=Path(__file__).parent.joinpath("data", "matrices_eigen"),
        fast=False,
    )


def copy_dataset(csv_path: Path, spath: Path, dpath: Path, suffix: str = ".npy"):

    df = pd.read_csv(csv_path)

    names = df["file_name"].tolist()
    for name in tqdm(names):
        shutil.copy(spath.joinpath(name + suffix), dpath.joinpath(name + suffix))


def main():
    for dir in ["train", "val", "test"]:
        copy_dataset(
            csv_path=Path(__file__).parent.joinpath(
                "data", "training_dataset", dir, f"{dir}.csv"
            ),
            spath=Path(__file__).parent.joinpath("data", "matrices_eigen"),
            dpath=Path(__file__).parent.joinpath("data", "training_dataset", dir),
        )
        CoulombUtils.find_maximum_shape(
            folder=Path(__file__).parent.joinpath("data", "training_dataset", dir),
        )

    Utils.generate_num_atoms(
        dataset_path=Path(__file__).parent.joinpath("data", "training_dataset"),
        xyz_path=Path(__file__).parent.joinpath("data", "xyz_files"),
        format=".npy",
    )


if __name__ == "__main__":
    main()
