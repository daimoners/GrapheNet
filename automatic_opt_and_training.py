try:

    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="dataset")
def main(cfg):
    for target in cfg.features:

        Utils.update_yaml(
            spath=Path().resolve().joinpath("config", "train_predict.yaml"),
            target_key="target",
            new_value=target,
        )

        print(f"Finding HP for target: {target}")
        process = subprocess.Popen(
            ["python", str(Path().resolve().joinpath("hp_finder.py"))]
        )
        process.wait()

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


if __name__ == "__main__":
    main()
