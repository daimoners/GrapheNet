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
    from ray.tune.suggest.optuna import OptunaSearch
    from pytorch_lightning import Trainer, seed_everything

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    seed_everything(42, workers=True)

    def train_tune(
        config,
        num_epochs=10,
        num_gpus=0,
    ):
        dataloaders = MyDataloader(
            cfg
        )  #! ho rimosso config perchè non mi serve il batch size
        model = MyRegressor(cfg, config)
        trainer = Trainer(
            deterministic=True,
            max_epochs=num_epochs,
            # If fractional GPUs passed in, convert to int.
            gpus=math.ceil(num_gpus),
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
            "lr": tune.loguniform(1e-4, 1e-3),
            # "batch_size": tune.choice([16, 32]),
        }

        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        reporter = CLIReporter(
            parameter_columns=["lr"],
            metric_columns=["loss", "mean_accuracy", "training_iteration"],
        )

        train_fn_with_parameters = tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
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
            name="tune_asha",
            search_alg=OptunaSearch(),
        )

        print("Best hyperparameters found were: ", analysis.best_config)

    tune_model_asha()


if __name__ == "__main__":
    main()
