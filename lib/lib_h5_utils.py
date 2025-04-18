try:

    import h5py
    import shutil
    from tqdm import tqdm
    import pandas as pd
    from pathlib import Path

except Exception as e:

    print(f"Some module are missing: {e}\n")


def get_atoms_name(atomic_number: int) -> str:

    atomic_name = {6: "C", 8: "O", 1: "H"}

    return atomic_name[atomic_number]


def translate_z(z_value: float) -> float:

    if z_value >= 5.0:
        return z_value - 10.0
    else:
        return z_value


def generate_xyz_dataset_from_h5(
    h5_dataset_path: Path, destination_path: Path, num_items: int = 0
) -> list:

    destination_path.mkdir(parents=True, exist_ok=True)

    items = len(
        [f for f in h5_dataset_path.iterdir() if f.suffix == ".h5" and f.is_file()]
    )

    list_files = []

    pbar = tqdm(total=num_items if num_items > 0 else items)
    for f in h5_dataset_path.iterdir():
        if f.suffix == ".h5":
            list_files.append(f.stem)
            with open(
                destination_path.joinpath(f.stem + ".xyz"),
                "w",
            ) as k:
                with h5py.File(f, "r") as f:
                    # get first object name/key; may or may NOT be a group
                    a_group_key = list(f.keys())

                    # get the object type for a_group_key: usually group or dataset
                    data = f[a_group_key[1]][
                        ()
                    ]  # to read a scalar dataspace (energy of the flake)

                    atoms_number = int(data.shape[0])

                    lines = []

                    lines.append(str(atoms_number) + "\n")
                    lines.append("Graphene\n")
                    for i in range(atoms_number):
                        lines.append(
                            str(get_atoms_name(int(data[i][0])))
                            + " "
                            + str(round(data[i][1], 5))
                            + " "
                            + str(round(data[i][2], 5))
                            + " "
                            + str(round(translate_z(data[i][3]), 5))
                            + "\n"
                        )

                k.writelines(lines)
            pbar.update(1)
            if num_items > 0 and pbar.n == num_items:
                break
    pbar.close()

    return list_files


def rename_and_move(spath: Path, dpath: Path, count: int = 1) -> int:

    dpath.mkdir(parents=True, exist_ok=True)

    for file in spath.iterdir():
        if file.suffix == ".h5":
            name = "graphene_" + str(count) + ".h5"
            shutil.copy(
                file,
                dpath.joinpath(name),
            )
            count += 1

    return count


def create_dataset_csv(
    h5_dataset_path: Path, destination_path: Path, from_list: list = None
):
    data = {
        "file_name": [],
        "a": [],
        "b": [],
        "c": [],
        "total_energy": [],
    }

    df = pd.DataFrame(data=data)

    if from_list is None:
        items = len(
            [f for f in h5_dataset_path.iterdir() if f.suffix == ".h5" and f.is_file()]
        )

        pbar = tqdm(total=items)
        for f in h5_dataset_path.iterdir():
            if f.suffix == ".h5":
                with h5py.File(f, "r") as h5_file:
                    # get first object name/key; may or may NOT be a group
                    a_group_key = list(h5_file.keys())

                    # get the object type for a_group_key: usually group or dataset
                    data = h5_file[a_group_key[0]][()]  # cell

                    a = data[0][0]
                    b = data[1][1]
                    c = data[2][2]

                    total_energy = h5_file[a_group_key[2]][()]

                    file_name = f.stem

                    df2 = pd.DataFrame(
                        [[file_name, a, b, c, total_energy]],
                        columns=["file_name", "a", "b", "c", "total_energy"],
                    )
                    df = pd.concat([df, df2], ignore_index=True)

                pbar.update(1)
        pbar.close()

    else:
        pbar = tqdm(total=len(from_list))
        for f in h5_dataset_path.iterdir():
            if f.stem in from_list:
                with h5py.File(f, "r") as h5_file:
                    # get first object name/key; may or may NOT be a group
                    a_group_key = list(h5_file.keys())

                    # get the object type for a_group_key: usually group or dataset
                    data = h5_file[a_group_key[0]][()]  # cell

                    a = data[0][0]
                    b = data[1][1]
                    c = data[2][2]

                    total_energy = h5_file[a_group_key[2]][()]

                    file_name = f.stem

                    df2 = pd.DataFrame(
                        [[file_name, a, b, c, total_energy]],
                        columns=["file_name", "a", "b", "c", "total_energy"],
                    )
                    df = pd.concat([df, df2], ignore_index=True)

                pbar.update(1)
        pbar.close()

    destination_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(destination_path.joinpath("dataset.csv"))


if __name__ == "__main__":
    pass
