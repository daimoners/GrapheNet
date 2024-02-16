try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    from lib.lib_utils import Utils
    import shutil
    import hydra
    from ray.tune.search.optuna import OptunaSearch
    import yaml
    from pathlib import Path
    import time
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything, Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

except Exception as e:
    print("Some module are missing {}".format(e))


class _TuneReportCallback(TuneReportCheckpointCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@hydra.main(version_base="1.2", config_path="config")
def main(cfg):
    if cfg.train.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif cfg.train.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    Path(__file__).parent.joinpath("ray_logs").mkdir(exist_ok=True, parents=True)

    seed_everything(42, workers=True)

    def train_tune(
        config,
        num_epochs=10,
    ):
        dataloaders = MyDataloader(cfg)
        model = MyRegressor(cfg, config)

        if cfg.cluster:
            trainer = Trainer(
                deterministic=cfg.deterministic,
                max_epochs=num_epochs,
                # If fractional GPUs passed in, convert to int.
                accelerator="gpu",
                num_nodes=1,
                devices=1,
                # strategy="ddp",
                logger=TensorBoardLogger(
                    save_dir=Path(__file__).parent.joinpath("ray_logs"),
                    name="",
                    version=".",
                ),
                callbacks=[
                    _TuneReportCallback(
                        {"loss": "val_loss", "mean_accuracy": "val_acc"},
                        on="validation_end",
                    )
                ],
                enable_progress_bar=False,
            )
        else:
            trainer = Trainer(
                deterministic=cfg.deterministic,
                max_epochs=num_epochs,
                # If fractional GPUs passed in, convert to int.
                accelerator="gpu",
                devices=1,
                logger=TensorBoardLogger(
                    save_dir=Path(__file__).parent.joinpath("ray_logs"),
                    name="",
                    version=".",
                ),
                callbacks=[
                    _TuneReportCallback(
                        {"loss": "val_loss", "mean_accuracy": "val_acc"},
                        on="validation_end",
                    )
                ],
                enable_progress_bar=False,
            )
        trainer.fit(model, datamodule=dataloaders)

    def tune_model_asha(
        num_samples=20,
        num_epochs=15,
        gpus_per_trial=1,
    ):
        config = {
            "lr": tune.loguniform(1e-4, 1),
        }

        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=10,
            reduction_factor=3,
        )

        reporter = CLIReporter(
            parameter_columns=["lr"],
            metric_columns=["loss", "mean_accuracy", "training_iteration"],
        )

        train_fn_with_parameters = tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        )
        resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

        start = time.time()
        analysis = tune.run(
            train_fn_with_parameters,
            resources_per_trial=resources_per_trial,
            metric="loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name=f"{cfg.target}_tune_asha",
            search_alg=OptunaSearch(),
        )
        end = time.time()

        results = {"optimization_time": float((end - start) / 60)}
        results.update(analysis.best_config)

        print("Best hyperparameters found were: ", analysis.best_config)

        dpath = Path(__file__).parent.joinpath(
            f"{cfg.train.dataset_path}/models/{cfg.target}"
        )
        dpath.mkdir(exist_ok=True, parents=True)

        with open(
            str(dpath.joinpath(f"{cfg.target}_optimization_results.yaml")),
            "w",
        ) as outfile:
            yaml.dump(results, outfile)

        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath("config", f"{cfg.name}.yaml"),
            target_key="base_lr",
            new_value=analysis.best_config["lr"],
        )
        Utils.update_yaml(
            spath=Path(__file__).parent.joinpath("config", f"{cfg.name}.yaml"),
            target_key=f"{cfg.target}",
            new_value=analysis.best_config["lr"],
        )

    if Path.home().joinpath("ray_results", f"{cfg.target}_tune_asha").is_dir():
        shutil.rmtree(
            str(Path.home().joinpath("ray_results", f"{cfg.target}_tune_asha"))
        )
    tune_model_asha()


if __name__ == "__main__":
    main()
