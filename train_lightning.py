try:
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    import hydra
    from pathlib import Path
    import time
    import yaml
    from torchsummary import summary
    import sys
    import io
    import torch
    from torch import _dynamo
    import subprocess
    from pytorch_lightning import seed_everything, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def write_results_yaml(cfg: dict, data: dict = None):
    dpath = Path(__file__).parent.joinpath(
        f"{cfg.train.dataset_path}/models/{cfg.target}"
    )
    dpath.mkdir(exist_ok=True, parents=True)
    if data is None:
        train_data = {
            "target": cfg.target,
            "num_epochs": cfg.train.num_epochs,
            "learning_rate": cfg.train.base_lr,
            "batch_size": cfg.train.batch_size,
            "dataset": cfg.train.dataset_path,
            "resolution": cfg.resolution,
        }
        with open(
            str(dpath.joinpath(f"{cfg.target}_train_results.yaml")), "w"
        ) as outfile:
            yaml.dump(train_data, outfile)
    else:
        with open(
            str(dpath.joinpath(f"{cfg.target}_train_results.yaml")), "a"
        ) as outfile:
            yaml.dump(data, outfile)


def save_model_summary(cfg: dict, model: MyRegressor):
    dpath = Path(__file__).parent.joinpath(
        f"{cfg.train.dataset_path}/models/{cfg.target}"
    )
    dpath.mkdir(exist_ok=True, parents=True)
    captured = io.StringIO()
    sys.stdout = captured
    summary(
        model.net.cuda(),
        (cfg.atom_types if not cfg.coulomb else 1, cfg.resolution, cfg.resolution),
        batch_size=cfg.train.batch_size,
        device="cuda",
    )
    sys.stdout = sys.__stdout__

    with open(str(dpath.joinpath("model_summary.txt")), "w") as f:
        # write the summary to the file
        f.write(captured.getvalue())


def get_model_name(model: MyRegressor):
    raw_name = str(type(model.net))
    chars_to_remove = "<>'"
    translate_table = str.maketrans("", "", chars_to_remove)
    name = raw_name.translate(translate_table)

    return name.split(".")[-1]


@hydra.main(version_base="1.2", config_path="config")
def main(cfg):
    if cfg.train.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif cfg.train.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    seed_everything(42, workers=True)

    model = MyRegressor(cfg)
    if cfg.train.compile:
        _dynamo.config.suppress_errors = True
        compiled_model = torch.compile(model)

    dpath = Path(__file__).parent.joinpath(
        f"{cfg.train.dataset_path}/models/{cfg.target}"
    )
    dpath.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(dpath),
        save_top_k=1,
        monitor="val_loss",
        filename="best_loss_{val_loss:.5f}_{epoch}",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=45, verbose=True, check_on_train_epoch_end=False
    )

    dataloaders = MyDataloader(cfg)

    if cfg.cluster:
        trainer = Trainer(
            deterministic=True,
            accelerator="gpu",
            num_nodes=1,
            devices=1,
            # strategy="ddp",
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                # model.get_progressbar(),
                early_stopping,
            ],
            enable_progress_bar=False,
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
        )

    write_results_yaml(cfg)
    write_results_yaml(cfg, data={"model_name": get_model_name(model)})
    save_model_summary(cfg, model)

    start = time.time()
    (
        trainer.fit(compiled_model, datamodule=dataloaders)
        if cfg.train.compile
        else trainer.fit(model, datamodule=dataloaders)
    )
    end = time.time()

    print(
        f"Completed training:\n TARGET = {cfg.target}\n DATASET = {cfg.train.dataset_path}\n NUM EPOCHS = {cfg.train.num_epochs}\n TRAINING TIME = {(end-start)/60:.3f} minutes"
    )

    write_results_yaml(cfg, data={"training_time": float((end - start) / 60)})

    process = subprocess.Popen(
        [
            "python",
            str(Path(__file__).parent.joinpath("predict_lightning.py")),
            "-cn",
            f"{cfg.name}.yaml",
        ]
    )
    process.wait()


if __name__ == "__main__":
    main()
