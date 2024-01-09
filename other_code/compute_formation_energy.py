try:
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from icecream import ic
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    import umap
    from lib.lib_utils import Utils
    from tqdm import tqdm

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def add_formation_energy_colummn(complete_csv_path: Path, dataset_path: Path):
    complete_csv = pd.read_csv(complete_csv_path)

    csv = pd.read_csv(dataset_path.joinpath("dataset.csv"))

    # Copia i valori della seconda colonna del primo CSV nella seconda colonna del secondo CSV
    csv["formation_energy"] = csv["file_name"].map(
        complete_csv.set_index("file_name")["formation_energy"]
    )
    csv["formation_energy"] = csv["formation_energy"].apply(
        lambda x: "{:.4f}".format(x)
    )

    # Salva il risultato nel secondo file CSV
    csv.to_csv(dataset_path.joinpath("dataset.csv"), index=False)

    for dir in ["train", "val", "test"]:
        csv = pd.read_csv(dataset_path.joinpath(dir, f"{dir}.csv"))

        csv["formation_energy"] = csv["file_name"].map(
            complete_csv.set_index("file_name")["formation_energy"]
        )
        csv["formation_energy"] = csv["formation_energy"].apply(
            lambda x: "{:.4f}".format(x)
        )

        # Salva il risultato nel secondo file CSV
        csv.to_csv(dataset_path.joinpath(dir, f"{dir}.csv"), index=False)


def add_total_energy_colummn(complete_csv_path: Path, dataset_path: Path):
    complete_csv = pd.read_csv(complete_csv_path)

    csv = pd.read_csv(dataset_path.joinpath("dataset.csv"))

    # Copia i valori della seconda colonna del primo CSV nella seconda colonna del secondo CSV
    csv["total_energy"] = csv["file_name"].map(
        complete_csv.set_index("file_name")["total_energy"]
    )
    csv["total_energy"] = csv["total_energy"].apply(lambda x: "{:.4f}".format(x))

    # Salva il risultato nel secondo file CSV
    csv.to_csv(dataset_path.joinpath("dataset.csv"), index=False)

    for dir in ["train", "val", "test"]:
        csv = pd.read_csv(dataset_path.joinpath(dir, f"{dir}.csv"))

        csv["total_energy"] = csv["file_name"].map(
            complete_csv.set_index("file_name")["total_energy"]
        )
        csv["total_energy"] = csv["total_energy"].apply(lambda x: "{:.4f}".format(x))

        # Salva il risultato nel secondo file CSV
        csv.to_csv(dataset_path.joinpath(dir, f"{dir}.csv"), index=False)


def add_energy_per_atom_column(dataset_path: Path):
    for dir in ["train", "val", "test"]:
        csv = pd.read_csv(dataset_path.joinpath(dir, f"{dir}.csv"))
        atoms = []
        samples = csv["file_name"].to_list()

        for sample in tqdm(samples):
            atoms.append(
                np.sum(np.loadtxt(dataset_path.joinpath(dir, f"{sample}.txt")))
            )

        csv["energy_per_atom"] = csv["total_energy"].div(atoms)
        csv["energy_per_atom"] = csv["energy_per_atom"].apply(
            lambda x: "{:.4f}".format(x)
        )

        csv.to_csv(dataset_path.joinpath(dir, f"{dir}.csv"), index=False)

        samples.clear()
        atoms.clear()


if __name__ == "__main__":
    # add_formation_energy_colummn(
    #     complete_csv_path=Path(
    #         "/home/cnrismn/git_workspace/GrapheNet/dataset_with_formation_energy.csv"
    #     ),
    #     dataset_path=Path(
    #         "/home/cnrismn/git_workspace/GrapheNet/test_other_networks/training_dataset"
    #     ),
    # )
    # add_total_energy_colummn(
    #     complete_csv_path=Path(
    #         "/home/tommaso/xyz_files_GO/dataset_with_formation_energy.csv"
    #     ),
    #     dataset_path=Path(
    #         "/home/tommaso/git_workspace/GrapheNet/Coulomb_GO/training_dataset"
    #     ),
    # )
    add_energy_per_atom_column(
        dataset_path=Path(
            "/home/tommaso/git_workspace/GrapheNet/data_GO/training_dataset_reduced"
        )
    )
