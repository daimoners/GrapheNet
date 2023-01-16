try:

    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pathlib import Path
    import time
    from pytorch_lightning.callbacks import RichProgressBar
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
    from pytorch_lightning import Trainer, seed_everything


except Exception as e:

    print("Some module are missing {}".format(e))


def get_progressbar():
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    return progress_bar


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):

    seed_everything(42, workers=True)

    model = MyRegressor(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(cfg.train.spath).joinpath("models", f"{cfg.target}")),
        save_top_k=1,
        monitor="val_loss",
        filename="best_loss_{val_loss:.5f}_{epoch}",
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(cfg.train.spath).joinpath("models")),
        save_top_k=1,
        monitor="epoch",
        mode="max",
        filename="model_{epoch}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=50, verbose=True, check_on_train_epoch_end=False
    )

    dataloaders = MyDataloader(cfg)

    if cfg.cluster:
        trainer = Trainer(
            deterministic=True,
            accelerator="gpu",
            num_nodes=1,
            devices=2,
            strategy="ddp",
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                get_progressbar(),
                early_stopping,
            ],
        )
    else:
        trainer = Trainer(
            deterministic=True,
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                get_progressbar(),
                early_stopping,
            ],
        )

    start = time.time()
    trainer.fit(model, dataloaders)
    end = time.time()

    print(
        f"Completed training:\n TARGET = {cfg.target}\n DATASET = {cfg.train.spath}\n NUM EPOCHS = {cfg.train.num_epochs}\n TRAINING TIME = {(end-start)/60:.3f} minutes"
    )


if __name__ == "__main__":
    main()
