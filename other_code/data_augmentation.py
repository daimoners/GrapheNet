try:
    from pathlib import Path
    import cv2
    from tqdm import tqdm
    from lib.lib_utils import Utils
    from PIL import Image
    import numpy as np
    import random
    import shutil
    import pandas as pd

except Exception as e:
    print(f"Some module are missing: {e}")


def crop_image(image: np.ndarray):
    if len(image.shape) == 2:
        image_data_bw = image
    else:
        image_data_bw = image.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    crop_box = (
        min(non_empty_rows),
        max(non_empty_rows),
        min(non_empty_columns),
        max(non_empty_columns),
    )

    if len(image.shape) == 2:
        image_data_new = image[
            crop_box[0] : crop_box[1] + 1, crop_box[2] : crop_box[3] + 1
        ]
    else:
        image_data_new = image[
            crop_box[0] : crop_box[1] + 1, crop_box[2] : crop_box[3] + 1, :
        ]

    return image_data_new


def main(dataset_path: Path):
    for dir in ["train", "val", "test"]:
        samples = [
            f
            for f in dataset_path.joinpath(dir).iterdir()
            if (f.suffix == ".png" and not "R" in f.stem)
        ]

        for sample in tqdm(samples):
            img = cv2.imread(str(sample))

            for angle in [1, 2, 3]:
                if angle == 1:
                    rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 2:
                    rotated_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 3:
                    rotated_image = cv2.rotate(img, cv2.ROTATE_180)

                cropped_image = crop_image(rotated_image)

                # Save the rotated image
                cv2.imwrite(
                    str(sample.with_stem(sample.stem + f"_R{angle}")), cropped_image
                )


def find_max_dims():
    path = Path(
        "/home/tommaso/git_workspace/GrapheNet/data_G/training_dataset_reduced/train"
    )
    samples = [f for f in path.iterdir() if f.suffix == ".png"]

    max_r = 0
    max_c = 0
    name_r = ""
    name_c = ""

    for sample in tqdm(samples):
        if "R" in sample.stem:
            continue
        img = cv2.imread(str(sample))
        rows, cols = img.shape[:2]
        if rows > max_r:
            max_r = rows
            name_r = sample.stem
        if cols > max_c:
            max_c = cols
            name_c = sample.stem

    print(f"Max columns: {max_c} with name {name_c}")
    print(f"Max rows: {max_r} with name {name_r}")


def copy_xyz_sample():
    spath = Path(
        "/home/tommaso/git_workspace/GrapheNet/data_G/training_dataset_reference/val"
    )
    dpath = Path(
        "/home/tommaso/git_workspace/GrapheNet/data_G/training_dataset_reduced/val"
    )

    samples = [f for f in spath.iterdir() if f.suffix == ".png"]

    for sample in tqdm(random.sample(samples, k=round(len(samples) / 4))):
        shutil.copy(sample, dpath.joinpath(sample.name))
        shutil.copy(sample.with_suffix(".txt"), dpath.joinpath(sample.stem + ".txt"))


def drop_from_pd():
    csv_path = Path(
        "/home/tommaso/git_workspace/GrapheNet/data_G/training_dataset_reduced/val/val.csv"
    )

    # create a sample dataframe
    df = pd.read_csv(csv_path)

    # create a list of values to drop
    samples = [
        f.stem
        for f in csv_path.parent.iterdir()
        if (f.suffix == ".png" and not "R" in f.stem)
    ]

    # drop columns based on values in column 'B'
    df = df[df["file_name"].isin(samples)]

    df.to_csv(csv_path)


def concatenate_ds(dataset_path: Path):
    df1 = pd.read_csv(dataset_path.joinpath("train", "train.csv"))
    df2 = pd.read_csv(dataset_path.joinpath("test", "test.csv"))
    df3 = pd.read_csv(dataset_path.joinpath("val", "val.csv"))

    # Concatena i tre DataFrame in uno solo
    result = pd.concat([df1, df2, df3], ignore_index=True)

    # Salva il DataFrame risultante in un nuovo file CSV
    result.to_csv(dataset_path.joinpath("dataset.csv"), index=False)


if __name__ == "__main__":
    # main(
    #     dataset_path=Path(
    #         "/home/tommaso/git_workspace/GrapheNet/data_GO/training_dataset_augmented"
    #     )
    # )
    find_max_dims()
