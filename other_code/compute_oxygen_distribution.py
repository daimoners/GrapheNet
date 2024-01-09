try:

    import pandas as pd
    from lib.lib_utils import Utils
    import numpy as np
    from pathlib import Path
    from scipy.spatial.distance import pdist
    from tqdm import tqdm

except Exception as e:

    print(f"Some module are missing for {__file__}: {e}\n")


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
    df = df.dropna(subset=[*targets, "band_gap"])

    indices = []
    for t in targets:
        idx = df.index[df[t] == 0.0].tolist()
        indices = [*indices, *idx]
    df = df.drop(indices, axis=0)

    return df


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


def main():
    # Iterate over all xyz files in the folder and compute the mean distribution of oxygen atoms for each file
    dataset_path = Path(
        "/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt/dataset.csv"
    )
    xyz_path = Path("/home/cnrismn/git_workspace/Chemception/data/xyz_files_opt")

    df = pd.read_csv(dataset_path)
    df = drop_nan_and_zeros(df)

    names = df["file_name"].to_list()

    distributions = []

    for name in tqdm(names):

        distributions.append(
            compute_mean_oxygen_distance(xyz_path.joinpath(name + ".xyz"))
        )

    distributions = (distributions - np.min(distributions)) / (
        np.max(distributions) - np.min(distributions)
    )

    df["distribution_mean"] = distributions
    df.to_csv(dataset_path.with_stem("dataset_complete"))


if __name__ == "__main__":
    main()
