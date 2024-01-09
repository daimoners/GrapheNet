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
                "Coulomb_G", "training_dataset", dir, f"{dir}.csv"
            ),
            spath=Path(__file__).parent.joinpath("Coulomb_G", "matrices_eigen"),
            dpath=Path(__file__).parent.joinpath("Coulomb_G", "training_dataset", dir),
        )
        CoulombUtils.find_maximum_shape(
            folder=Path(__file__).parent.joinpath("Coulomb_G", "training_dataset", dir),
        )

    Utils.generate_num_atoms(
        dataset_path=Path(__file__).parent.joinpath("Coulomb_G", "training_dataset"),
        xyz_path=Path(__file__).parent.joinpath("Coulomb_G", "xyz_files"),
        format=".npy",
    )


# def dummy():
#     df = pd.read_csv(
#         "/home/tommaso/git_workspace/GrapheNet/data_G/training_dataset_reference/dataset_with_energy_per_atom_new.csv"
#     )

#     samples = df["file_name"].to_list()

#     for sample in tqdm(samples):
#         shutil.copy(
#             f"/home/tommaso/xyz_files_G/{sample}.xyz",
#             f"/home/tommaso/git_workspace/GrapheNet/Coulomb_G/xyz_files/{sample}.xyz",
#         )


if __name__ == "__main__":
    main()
