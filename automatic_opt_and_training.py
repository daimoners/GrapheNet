try:

    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path
    from telegram_bot import send_images

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    for target in list(cfg.train.lr_list.keys()):

        Utils.update_yaml(
            spath=Path().resolve().joinpath("config", "train_predict.yaml"),
            target_key="target",
            new_value=target,
        )

        if cfg.train.base_lr > 0.0:
            print(f"Finding HP for target: {target}")
            process = subprocess.Popen(
                ["python", str(Path().resolve().joinpath("hp_finder.py"))]
            )
            process.wait()
        else:
            Utils.update_yaml(
                spath=Path().resolve().joinpath("config", "train_predict.yaml"),
                target_key="base_lr",
                new_value=cfg.train.lr_list[target],
            )

        print(f"Training for target: {target}")
        process = subprocess.Popen(
            ["python", str(Path().resolve().joinpath("train_lightning.py"))]
        )
        process.wait()

        print(f"Prediction for target: {target}\n")
        process = subprocess.Popen(
            ["python", str(Path().resolve().joinpath("predict_lightning.py"))]
        )
        process.wait()

    Utils.update_yaml(
        spath=Path().resolve().joinpath("config", "train_predict.yaml"),
        target_key="base_lr",
        new_value=0.0,
    )

    images_paths = []
    captions = []
    for target in list(cfg.train.lr_list.keys()):
        images_paths.append(
            Path(cfg.train.spath).joinpath("models", str(target), f"{target}_fit.png")
        )
        captions.append(
            str(
                Path(cfg.train.spath).joinpath(
                    "models", str(target), f"{target}_fit.png"
                )
            )
        )

    send_images(images_paths, captions)


if __name__ == "__main__":
    main()
