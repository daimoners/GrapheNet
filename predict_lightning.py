try:
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from lightning import Trainer, seed_everything
    from pathlib import Path
    import numpy as np
    from lib.lib_utils import Utils
    import yaml
    import torch

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def get_checkpoint_name(checkpoints_path: Path):
    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    return str(best_loss[0])


@hydra.main(version_base="1.2", config_path="config")
def main(cfg):
    if cfg.train.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif cfg.train.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    seed_everything(42, workers=True)

    dpath = Path(__file__).parent.joinpath(
        f"{cfg.train.dataset_path}/models/{cfg.target}"
    )
    dpath.mkdir(exist_ok=True, parents=True)

    checkpoints = get_checkpoint_name(dpath)

    model = MyRegressor(cfg)

    dataloaders = MyDataloader(cfg)

    trainer = Trainer(
        deterministic=cfg.deterministic,
        accelerator="gpu",
        devices=1,
        callbacks=[model.get_progressbar()],
    )

    trainer.test(
        model,
        datamodule=dataloaders,
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
            dpath.joinpath(
                f"{cfg.target}_prediction_results.yaml",
            )
        ),
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)

    Utils.plot_fit(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        dpath=dpath.joinpath(f"{cfg.target}_fit.png"),
        target=cfg.target,
    )

    Utils.write_csv_results(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        names=model.sample_names,
        dpath=dpath.joinpath(f"{cfg.target}_prediction_results.csv"),
        target=cfg.target,
    )


if __name__ == "__main__":
    main()
