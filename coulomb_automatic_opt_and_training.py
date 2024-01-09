try:
    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path
    from telegram_bot import send_images

except Exception as e:
    print("Some module are missing {}".format(e))


@hydra.main(
    version_base="1.2", config_path="config", config_name="train_predict_coulomb"
)
def main(cfg):
    for target in list(cfg.train.lr_list.keys()):
        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath(
                "config", "train_predict_coulomb.yaml"
            ),
            target_key="target",
            new_value=target,
        )

        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath(
                "config", "train_predict_coulomb.yaml"
            ),
            target_key="base_lr",
            new_value=cfg.train.lr_list[target],
        )

        print(f"Training for target: {target}")
        process = subprocess.Popen(
            [
                "python",
                str(Path(__file__).parent.joinpath("coulomb_train_lightning.py")),
            ]
        )
        process.wait()

    Utils.update_yaml(
        spath=Path(__file__).parent.joinpath("config", "train_predict_coulomb.yaml"),
        target_key="base_lr",
        new_value=0.0,
    )

    images = {}
    for target in list(cfg.train.lr_list.keys()):
        images[
            str(
                Path(cfg.train.spath).joinpath(
                    "models", str(target), f"{target}_fit.png"
                )
            )
        ] = Path(cfg.train.spath).joinpath("models", str(target), f"{target}_fit.png")

    send_images(images)


if __name__ == "__main__":
    main()
