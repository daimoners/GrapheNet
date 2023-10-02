try:
    from pathlib import Path
    import yaml
    from tabulate import tabulate
    from argparse import ArgumentParser

except Exception as e:
    print(f"Some moduel are missign from {__file__}: {e}\n")


def main(
    dataset_paths: [Path],
    targets: list = [
        "Fermi_energy",
        "electron_affinity",
        "electronegativity",
        "ionization_potential",
        "formation_energy",
    ],
    only_mean: bool = False,
):
    first_table = []
    for n, path in enumerate(dataset_paths):
        first_table.append([n, str(path)])
    table = tabulate(first_table, headers="firstrow", tablefmt="grid")
    print("\n")
    print(table)
    print("\n")

    for target in targets:
        max_error = ["Maximum % error"]
        mean_error = ["Mean % error"]
        std_error = ["STD % error"]
        first_row = [target]
        for n, path in enumerate(dataset_paths):
            results = path.joinpath(target, f"{target}_prediction_results.yaml")

            with open(results, "r") as file:
                data = yaml.safe_load(file)

            first_row.append(n)
            max_error.append(data.get("Maximum % error"))
            mean_error.append(data.get("Mean % error"))
            std_error.append(data.get("STD % error"))

        if only_mean:
            data_to_compare = [
                first_row,
                mean_error,
            ]
        else:
            data_to_compare = [
                first_row,
                max_error,
                mean_error,
                std_error,
            ]

        table = tabulate(data_to_compare, headers="firstrow", tablefmt="grid")
        print(table)

        first_row.clear()
        max_error.clear()
        mean_error.clear()
        std_error.clear()


if __name__ == "__main__":
    parser = ArgumentParser(description="Compare the results of 2 datasets")
    parser.add_argument("paths", nargs="+", type=str, help="Dataset paths")
    parser.add_argument(
        "--mean",
        "-m",
        action="store_true",
        help="Compare only mean value",
    )
    args = parser.parse_args()

    paths = []

    for path in args.paths:
        paths.append(Path(path))

    main(paths, only_mean=args.mean)
