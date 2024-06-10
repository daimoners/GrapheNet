try:
    import os
    import shutil
    import argparse

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def clone_files_from_dirs(source_dir, dest_dir, extensions: tuple):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    count = 0

    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)
        dest_item_path = os.path.join(dest_dir, item)

        if os.path.isdir(source_item_path):
            count += clone_files_from_dirs(source_item_path, dest_item_path, extensions)

        elif os.path.isfile(source_item_path) and item.lower().endswith(extensions):
            shutil.copyfile(source_item_path, dest_item_path)
            count += 1

    return count


def remove_empty_dirs(directory, verbose: bool = False):

    for root, dirs, files in os.walk(directory, topdown=False):
        if not dirs and not files:
            os.rmdir(root)

        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except FileNotFoundError:
                if verbose:
                    print(f"Warning, directory {dir_path} not found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clone a directory with subdirs copying only the items with a desired suffix."
    )
    parser.add_argument("source_folder", type=str, help="Source folder.")
    parser.add_argument("destination_folder", type=str, help="Destination folder.")
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=[".csv", ".yaml", ".txt", ".ipynb"],
        help="List of file extensions to copy (e.g. -e png jpg).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbosity.",
    )
    args = parser.parse_args()

    count = clone_files_from_dirs(
        args.source_folder, args.destination_folder, tuple(args.extensions)
    )
    remove_empty_dirs(args.destination_folder, args.verbose)

    with open(
        str(os.path.join(args.destination_folder, "number_of_files_copied.txt")), "w"
    ) as f:
        f.write(f"{count} files were copied!\n")
        f.write(f"The filtered extensions are: {args.extensions}\n")
