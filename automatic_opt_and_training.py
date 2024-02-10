try:
    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path

except Exception as e:
    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config")
def main(cfg):
    name_stem = "train_predict_coulomb" if cfg.coulomb else "train_predict"

    for target in list(cfg.train.lr_list.keys()):
        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath("config", f"{name_stem}.yaml"),
            target_key="target",
            new_value=target,
        )

        if cfg.train.base_lr > 0.0:
            print(f"Finding HP for target: {target}")
            process = subprocess.Popen(
                ["python", str(Path(__file__).parent.joinpath("hp_finder.py"))]
            )
            process.wait()

        else:
            Utils.update_yaml(
                spath=Path(__file__).parent.joinpath("config", f"{name_stem}.yaml"),
                target_key="base_lr",
                new_value=cfg.train.lr_list[target],
            )

        print(f"Training for target: {target}")
        process = subprocess.Popen(
            ["python", str(Path(__file__).parent.joinpath("train_lightning.py"))]
        )
        process.wait()

    Utils.update_yaml(
        spath=Path(__file__).parent.joinpath("config", f"{name_stem}.yaml"),
        target_key="base_lr",
        new_value=0.0,
    )


if __name__ == "__main__":
    main()
