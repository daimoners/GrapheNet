try:
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    from lib.lib_networks import MyDatasetPng
    from lib.lib_utils import Utils
    import hydra
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
    from sklearn.model_selection import KFold
    import pandas as pd
    from torch.utils.data import DataLoader
    from omegaconf import open_dict
    import numpy as np
    import yaml
    import math
    import submitit
    from icecream import ic
    from omegaconf import open_dict
    from telegram_bot import send_message


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
            "resolution": cfg.resolution,
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


def get_checkpoint_name(checkpoints_path: Path):
    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ]

    return str(best_loss[0])


def get_kfold_results(folds_path: Path, target: str):
    folds = [
        f for f in folds_path.iterdir() if (f.is_dir() and "fold" in f.name.lower())
    ]

    max = []
    mean = []
    std = []

    for fold in folds:
        with open(str(fold.joinpath(f"{target}_prediction_results.yaml")), "r") as file:
            config = yaml.safe_load(file)

        max.append(config["Maximum % error"])
        mean.append(config["Mean % error"])
        std.append(config["STD % error"])

    return np.mean(max), np.mean(mean), np.mean(std)


@hydra.main(version_base="1.2", config_path="config", config_name="train_predict")
def main(cfg):
    if cfg.train.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif cfg.train.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    with open_dict(cfg):
        cfg.train.base_lr = cfg.train.lr_list[cfg.target]

    seed_everything(42, workers=True)

    # early_stopping = EarlyStopping(
    #     monitor="val_loss", patience=45, verbose=True, check_on_train_epoch_end=False
    # )

    df = pd.read_csv(Path(cfg.train.spath).joinpath("dataset.csv"))
    # folds = round(1 + math.log2(len(df)))
    folds = 6
    kfold = KFold(n_splits=folds, shuffle=True)
    images_path = Path(cfg.train.spath).joinpath("images")

    # test_df = pd.read_csv(Path(cfg.train.spath).joinpath("test", "test.csv"))
    # test_samples = test_df["file_name"].to_list()
    # test_images = [
    #     f
    #     for f in Path(cfg.train.spath).joinpath("test").iterdir()
    #     if (f.suffix.lower() == ".png" and f.stem in test_samples)
    # ]
    # test_data = MyDatasetPng(
    #     test_images,
    #     test_df,
    #     cfg.target,
    #     resolution=cfg.resolution,
    #     enlargement_method=cfg.enlargement_method,
    #     phase="test",
    # )
    # test_dataloader = DataLoader(
    #     test_data,
    #     batch_size=cfg.train.batch_size,
    #     shuffle=False,
    #     num_workers=(cfg.num_workers if not cfg.cluster else cfg.cluster_num_workers),
    #     pin_memory=True,
    #     drop_last=True,
    # )

    original_dpath = Path(cfg.train.dpath)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(df)):
        model = MyRegressor(cfg)
        if cfg.train.compile:
            compiled_model = torch.compile(model)

        with open_dict(cfg):
            cfg.train.dpath = str(original_dpath.joinpath(f"fold_{fold}"))

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.train.dpath,
            save_top_k=1,
            monitor="train_acc",
            mode="max",
            filename="best_loss_{epoch}",
        )
        checkpoint_callback_every_n_epochs = ModelCheckpoint(
            dirpath=cfg.train.dpath,
            every_n_epochs=25,
            save_top_k=1,
            filename="best_loss_{epoch}",
        )

        train_df = df.iloc[train_ids]
        val_df = df.iloc[val_ids]

        train_samples = train_df["file_name"].to_list()
        val_samples = val_df["file_name"].to_list()

        train_images = [
            f
            for f in images_path.iterdir()
            if (f.suffix.lower() == ".png" and f.stem in train_samples)
        ]
        val_images = [
            f
            for f in images_path.iterdir()
            if (f.suffix.lower() == ".png" and f.stem in val_samples)
        ]

        train_data = MyDatasetPng(
            train_images,
            train_df,
            cfg.target,
            resolution=cfg.resolution,
            enlargement_method=cfg.enlargement_method,
            phase="train",
        )
        val_data = MyDatasetPng(
            val_images,
            val_df,
            cfg.target,
            resolution=cfg.resolution,
            enlargement_method=cfg.enlargement_method,
            phase="test",
        )

        train_dataloader = DataLoader(
            train_data,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=(
                cfg.num_workers if not cfg.cluster else cfg.cluster_num_workers
            ),
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=(
                cfg.num_workers if not cfg.cluster else cfg.cluster_num_workers
            ),
            pin_memory=True,
            drop_last=True,
        )

        if cfg.cluster:
            trainer = Trainer(
                deterministic=True,
                accelerator="gpu",
                num_nodes=1,
                devices=1,
                # strategy="ddp",
                max_epochs=cfg.train.num_epochs,
                callbacks=[
                    checkpoint_callback_every_n_epochs,
                    # model.get_progressbar(),
                    # early_stopping,
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
                    checkpoint_callback_every_n_epochs,
                    model.get_progressbar(),
                    # early_stopping,
                ],
            )

        write_results_yaml(cfg)
        write_results_yaml(cfg, data={"model_name": get_model_name(model)})
        save_model_summary(cfg, model)

        start = time.time()
        (
            trainer.fit(compiled_model, train_dataloaders=train_dataloader)
            if cfg.train.compile
            else trainer.fit(model, train_dataloaders=train_dataloader)
        )
        end = time.time()

        write_results_yaml(
            cfg,
            data={
                "training_time": float((end - start) / 60),
                "train_loss": model.train_loss_plot,
                "train_acc": model.train_acc_plot,
                # "val_loss": model.val_loss_plot,
                # "val_acc": model.val_acc_plot,
            },
        )
        Utils.plot_loss_acc(
            values=model.train_loss_plot,
            dpath=Path(cfg.train.dpath).joinpath(f"train_loss.png"),
            type="Train Loss",
        )
        Utils.plot_loss_acc(
            values=model.train_acc_plot,
            dpath=Path(cfg.train.dpath).joinpath(f"train_acc.png"),
            type="Train Acc",
        )
        # Utils.plot_loss_acc(
        #     values=model.val_loss_plot,
        #     dpath=Path(cfg.train.dpath).joinpath(f"val_loss.png"),
        #     type="Val Loss",
        # )
        # Utils.plot_loss_acc(
        #     values=model.val_acc_plot,
        #     dpath=Path(cfg.train.dpath).joinpath(f"val_acc.png"),
        #     type="Val Acc",
        # )

        checkpoints = get_checkpoint_name(Path(cfg.train.dpath))
        trainer.test(model, val_dataloader, ckpt_path=checkpoints)
        performance = {
            "Maximum % error": float(np.max(model.errors)),
            "Mean % error": float(np.mean(model.errors)),
            "STD % error": float(np.std(model.errors)),
        }
        with open(
            str(
                Path(cfg.train.dpath).joinpath(
                    f"{cfg.target}_prediction_results.yaml",
                )
            ),
            "w",
        ) as outfile:
            yaml.dump(performance, outfile)

        Utils.plot_fit(
            y=model.plot_y,
            y_hat=model.plot_y_hat,
            dpath=Path(cfg.train.dpath).joinpath(f"{cfg.target}_fit.png"),
            target=cfg.target,
        )

        Utils.write_csv_results(
            y=model.plot_y,
            y_hat=model.plot_y_hat,
            names=model.sample_names,
            dpath=Path(cfg.train.dpath).joinpath(
                f"{cfg.target}_prediction_results.csv"
            ),
            target=cfg.target,
        )

    max, mean, std = get_kfold_results(original_dpath, cfg.target)
    data = {"max_kfold": float(max), "mean_kfold": float(mean), "std_kfold": float(std)}
    with open(
        str(original_dpath.joinpath(f"{cfg.target}_kfold_prediction_results.yaml")),
        "w",
    ) as outfile:
        yaml.dump(data, outfile, indent=4)

    message = f"Prediction on target `{cfg.target}` completed for Kfolds ✅"
    send_message(message, parse_mode="MarkdownV2")


if __name__ == "__main__":
    main()
