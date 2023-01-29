try:

    from pytorch_lightning.loggers import TensorBoardLogger
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.integration.pytorch_lightning import (
        TuneReportCallback,
        TuneReportCheckpointCallback,
    )
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    from pytorch_lightning import Trainer
    import math
    import hydra
    from ray.tune.search.optuna import OptunaSearch
    from pytorch_lightning import Trainer, seed_everything
    import yaml
    from pathlib import Path

except Exception as e:

    print("Some module are missing {}".format(e))


def update_yaml(new_value, target_key="base_lr"):
    def search_and_modify(data, target_key, new_value):
        for key, value in data.items():
            if key == target_key:
                data[key] = new_value
                return True
            elif type(value) is dict:
                if search_and_modify(value, target_key, new_value):
                    return True
        return False

    # Open the YAML file
    with open(
        str(Path().resolve().joinpath("config", "train_predict.yaml")), "r"
    ) as file:
        # Load the YAML data into a Python dictionary
        data = yaml.load(file, Loader=yaml.FullLoader)

    if search_and_modify(data, target_key, new_value):
        with open(
            str(Path().resolve().joinpath("config", "train_predict.yaml")), "w"
        ) as file:
            yaml.dump(data, file)
        print(f"{target_key} value changed to {new_value}")
    else:
        print(f"{target_key} not found in the file")


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    seed_everything(42, workers=True)

    def train_tune(
        config,
        num_epochs=10,
    ):
        dataloaders = MyDataloader(
            cfg
        )  #! ho rimosso config perchè non mi serve il batch size
        model = MyRegressor(cfg, config)
        if cfg.cluster:
            trainer = Trainer(
                deterministic=True,
                max_epochs=num_epochs,
                # If fractional GPUs passed in, convert to int.
                accelerator="gpu",
                num_nodes=1,
                devices=2,
                strategy="ddp",
                logger=TensorBoardLogger(
                    save_dir=tune.get_trial_dir(), name="", version="."
                ),
                callbacks=[
                    TuneReportCallback(
                        {"loss": "val_loss", "mean_accuracy": "val_acc"},
                        on="validation_end",
                    )
                ],
            )
        else:
            trainer = Trainer(
                deterministic=True,
                max_epochs=num_epochs,
                # If fractional GPUs passed in, convert to int.
                accelerator="gpu",
                devices=1,
                logger=TensorBoardLogger(
                    save_dir=tune.get_trial_dir(), name="", version="."
                ),
                callbacks=[
                    TuneReportCallback(
                        {"loss": "val_loss", "mean_accuracy": "val_acc"},
                        on="validation_end",
                    )
                ],
            )
        trainer.fit(model, dataloaders)

    def tune_model_asha(
        num_samples=20,
        num_epochs=15,
        gpus_per_trial=1,
    ):
        config = {
            "lr": tune.loguniform(
                1e-4, 1e-1
            ),  # dopo aver trovato un certo range con "lr": tune.loguniform(1e-4, 1e-2) e analizzando i risultati su tensorboard, uso "lr": tune.loguniform(0.007, 0.009) sul range trovato con il range (1e-4, 1e-2)
            # "batch_size": tune.choice([16, 32]),
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

        print("Best hyperparameters found were: ", analysis.best_config)
        with open(
            str(Path(cfg.train.spath).joinpath(f"{cfg.target}_best_config.yaml")), "w"
        ) as outfile:
            yaml.dump(analysis.best_config, outfile)
        update_yaml(new_value=analysis.best_config["lr"])

    tune_model_asha()


if __name__ == "__main__":
    main()
