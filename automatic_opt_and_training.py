try:
    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path

except Exception as e:
    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config")
def main(cfg):

    for target in list(cfg.train.lr_list.keys()):
        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath("config", f"{cfg.name}.yaml"),
            target_key="target",
            new_value=target,
        )

        if (
            isinstance(cfg.train.lr_list[target], float)
            and cfg.train.lr_list[target] > 0.0
        ):
            Utils.update_yaml(
                spath=Path(__file__).parent.joinpath("config", f"{cfg.name}.yaml"),
                target_key="base_lr",
                new_value=cfg.train.lr_list[target],
            )

        else:
            print(f"Finding HP for target: {target}")
            process = subprocess.Popen(
                [
                    "python",
                    str(Path(__file__).parent.joinpath("hp_finder.py")),
                    "-cn",
                    f"{cfg.name}.yaml",
                ]
            )
            process.wait()

        print(f"Training for target: {target}")
        process = subprocess.Popen(
            [
                "python",
                str(Path(__file__).parent.joinpath("train_lightning.py")),
                "-cn",
                f"{cfg.name}.yaml",
            ]
        )
        print(process.args)
        process.wait()


if __name__ == "__main__":
    main()
