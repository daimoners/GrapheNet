try:
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.tuner import Tuner
    from pathlib import Path
    import time
    from lightning import Trainer, seed_everything
    import yaml
    from torchsummary import summary
    import sys
    import io
    import torch
    from datetime import datetime
    import subprocess


except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def write_results_yaml(cfg: dict, data: dict = None):
    Path(cfg.train.dpath).mkdir(exist_ok=True, parents=True)
    if data is None:
        train_data = {
            "target": cfg.target,
            "num_epochs": cfg.train.num_epochs,
            "learning_rate": cfg.train.base_lr,
            "batch_size": cfg.train.batch_size,
            "dataset": cfg.train.spath,
        }
        with open(
            str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_train_results.yaml")), "w"
        ) as outfile:
            yaml.dump(train_data, outfile)
    else:
        with open(
            str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_train_results.yaml")), "a"
        ) as outfile:
            yaml.dump(data, outfile)


def save_model_summary(cfg: dict, model: MyRegressor):
    captured = io.StringIO()
    sys.stdout = captured
    summary(
        model.net.cuda(),
        (cfg.atom_types if not cfg.coulomb else 1, cfg.resolution, cfg.resolution),
        batch_size=cfg.train.batch_size,
        device="cuda",
    )
    sys.stdout = sys.__stdout__

    with open(str(Path(cfg.train.dpath).joinpath("model_summary.txt")), "w") as f:
        # write the summary to the file
        f.write(captured.getvalue())


def get_model_name(model: MyRegressor):
    raw_name = str(type(model.net))
    chars_to_remove = "<>'"
    translate_table = str.maketrans("", "", chars_to_remove)
    name = raw_name.translate(translate_table)

    return name.split(".")[-1]


def set_logger(cfg):
    now = datetime.now()
    now_string = now.strftime("%b-%d-%Y_%H:%M:%S")

    logger = WandbLogger(
        project=cfg.logger.project_name,
        log_model=True,
        name=f"{cfg.target}_{now_string}",
    )

    return logger, f"{cfg.target}_{now_string}"


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    if cfg.train.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif cfg.train.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    seed_everything(42, workers=True)

    model = MyRegressor(cfg)
    if cfg.train.compile:
        compiled_model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.dpath,
        save_top_k=1,
        monitor="val_loss",
        filename="best_loss_{val_loss:.5f}_{epoch}",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=35, verbose=True, check_on_train_epoch_end=False
    )

    dataloaders = MyDataloader(cfg)

    wandb_logger, experiment_name = set_logger(cfg)

    if cfg.cluster:
        trainer = Trainer(
            deterministic=cfg.deterministic,
            accelerator="gpu",
            num_nodes=1,
            devices=2,
            strategy="ddp",
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                model.get_progressbar(),
                early_stopping,
            ],
            logger=wandb_logger,
        )
    else:
        trainer = Trainer(
            deterministic=cfg.deterministic,
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                model.get_progressbar(),
                early_stopping,
            ],
            logger=wandb_logger,
        )

    write_results_yaml(cfg)
    write_results_yaml(cfg, data={"model_name": get_model_name(model)})
    save_model_summary(cfg, model)

    # tuner = Tuner(trainer)
    # # Run learning rate finder
    # lr_finder = tuner.lr_find(
    #     model,
    #     dataloaders,
    #     min_lr=1e-5,
    #     max_lr=0.1,
    # )
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # # update hparams of the model
    # model.learning_rate = new_lr

    start = time.time()
    trainer.fit(compiled_model, dataloaders) if cfg.train.compile else trainer.fit(
        model, dataloaders
    )
    end = time.time()

    print(
        f"Completed training:\n TARGET = {cfg.target}\n DATASET = {cfg.train.spath}\n NUM EPOCHS = {cfg.train.num_epochs}\n TRAINING TIME = {(end-start)/60:.3f} minutes"
    )

    write_results_yaml(cfg, data={"training_time": float((end - start) / 60)})

    process = subprocess.Popen(
        ["python", str(Path().resolve().joinpath("predict_lightning.py"))]
    )
    process.wait()

    trainer.logger.log_image(
        key="fit",
        images=[str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_fit.png"))],
        caption=[f"{cfg.target}_fit.png"],
    )

    with open(
        str(Path(cfg.train.dpath).joinpath(f"{cfg.target}_prediction_results.yaml")),
        "r",
    ) as file:
        results = yaml.safe_load(file)
    results["model_name"] = get_model_name(model)

    for key, value in results.items():
        if not isinstance(value, str):
            results[key] = str(value)

    trainer.logger.log_text(
        key=f"prediction_results_{experiment_name}",
        columns=list(results.keys()),
        data=[list(results.values())],
    )


if __name__ == "__main__":
    main()
