try:

    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from pytorch_lightning import Trainer, seed_everything
    from pathlib import Path
    import numpy as np

except Exception as e:

    print("Some module are missing {}".format(e))


def get_model_names(checkpoints_path: Path):

    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    last = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("model_epoch")
    ]

    return str(best_loss[0]), str(last[0])


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):

    seed_everything(42, workers=True)

    checkpoints = get_model_names(Path(__file__).parent.joinpath("models"))

    model = MyRegressor(cfg)

    dataloaders = MyDataloader(cfg)

    trainer = Trainer(
        deterministic=True,
        accelerator="gpu",
        devices=1,
    )

    trainer.test(
        model,
        dataloaders,
        ckpt_path=checkpoints[0],
    )
    print(len(model.errors))
    print("Maximum % error = {:.5f}%".format(np.max(model.errors)))
    print("Mean % error = {:.5f}%\n".format(np.mean(model.errors)))

    trainer.test(
        model,
        dataloaders,
        ckpt_path=checkpoints[1],
    )
    print(len(model.errors))
    print("Maximum % error = {:.5f}%".format(np.max(model.errors)))
    print("Mean % error = {:.5f}%\n".format(np.mean(model.errors)))


if __name__ == "__main__":
    main()
