try:
    import subprocess
    from lib.lib_utils import Utils
    import hydra
    from pathlib import Path

except Exception as e:
    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    for target in list(cfg.train.lr_list.keys()):
        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath("config", "train_predict.yaml"),
            target_key="target",
            new_value=target,
        )

        print(f"Training for target: {target}")
        process = subprocess.Popen(
            ["python", str(Path(__file__).parent.joinpath("train_lightning.py"))]
        )
        process.wait()


if __name__ == "__main__":
    main()
