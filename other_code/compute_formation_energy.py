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


if __name__ == "__main__":
    add_formation_energy_colummn(
        complete_csv_path=Path(
            "/home/cnrismn/git_workspace/GrapheNet/dataset_with_formation_energy.csv"
        ),
        dataset_path=Path(
            "/home/cnrismn/git_workspace/GrapheNet/test_other_networks/training_dataset"
        ),
    )
