try:

    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from pytorch_lightning import Trainer, seed_everything
    from pathlib import Path
    import numpy as np
    from lib.lib_utils import Utils
    import yaml

except Exception as e:

    print("Some module are missing {}".format(e))


def get_model_names(checkpoints_path: Path):

    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    return str(best_loss[0])


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):

    seed_everything(42, workers=True)

    checkpoints = get_model_names(Path(cfg.train.dpath))

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
        ckpt_path=checkpoints,
    )
    print("Maximum % error = {:.5f}%".format(np.max(model.errors)))
    print("Mean % error = {:.5f}%".format(np.mean(model.errors)))
    print("STD % error = {:.5f}%\n".format(np.std(model.errors)))

    performance = {
        "Maximum % error": float(np.max(model.errors)),
        "Mean % error": float(np.mean(model.errors)),
        "STD % error": float(np.std(model.errors)),
    }

    with open(
        str(
            Path(cfg.train.dpath).joinpath(
                f"{cfg.target}_best_results.yaml",
            )
        ),
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)

    Utils.plot_fit(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        dpath=Path(cfg.train.dpath).joinpath(f"{cfg.target}_best_fit.png"),
        target=cfg.target,
    )


if __name__ == "__main__":
    main()
