try:
    from lib.lib_trainer_predictor_lightning import MyRegressor, MyDataloader
    from lib.lib_utils import Utils
    import hydra
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from pathlib import Path
    import time
    from lightning import Trainer, seed_everything
    import yaml
    from torchsummary import summary
    import sys
    import io
    import torch
    import submitit
    from icecream import ic
    from omegaconf import open_dict
    import numpy as np


except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_GrapheNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        start(self.cfg)


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
    try:
        summary(
            model.net.cuda(),
            (cfg.atom_types if not cfg.coulomb else 1, cfg.resolution, cfg.resolution),
            batch_size=cfg.train.batch_size,
            device="cuda",
        )
    except:
        summary(
            model.net.cuda(),
            (cfg.atom_types if not cfg.coulomb else 1, cfg.resolution**2),
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


def get_last_checkpoint_name(checkpoints_path: Path):
    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("last")
    ]

    return str(best_loss[0])


def get_difference_checkpoint_name(checkpoints_path: Path):
    best_loss = [
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("loss_difference")
    ]

    return str(best_loss[0])


@hydra.main(
    version_base="1.2", config_path="config", config_name="train_predict_coulomb"
)
def main(cfg):
    if cfg.verbose:
        ic.enable()
    else:
        ic.disable()

    for target in list(cfg.train.lr_list.keys()):
        if Path(cfg.train.dpath).parent.joinpath(target).is_dir():
            continue

        with open_dict(cfg):
            cfg.target = target
            cfg.train.base_lr = cfg.train.lr_list[target]

        executor = submitit.AutoExecutor(
            folder=str(Path(cfg.slurm_output)),
            slurm_max_num_timeout=30,
        )

        executor.update_parameters(
            mem_gb=0 if not cfg.slurm_mem else cfg.slurm_mem,
            gpus_per_node=0 if not cfg.slurm_ngpus else cfg.slurm_ngpus,
            tasks_per_node=1 if not cfg.slurm_ngpus else cfg.slurm_ngpus,
            cpus_per_task=2 if not cfg.slurm_ncpus else cfg.slurm_ncpus,
            timeout_min=cfg.slurm_timeout,
            slurm_partition=cfg.slurm_partition,
            slurm_exclude=cfg.slurm_exclude,
        )

        if cfg.slurm_nodelist:
            executor.update_parameters(
                slurm_additional_parameters={"nodelist": f"{cfg.slurm_nodelist}"}
            )

        executor.update_parameters(name=f"{cfg.slurm_job_name}")
        slurm_auto_dftb = SLURM_GrapheNet(cfg)
        job = executor.submit(slurm_auto_dftb)
        print(f"Submitted job_id: {job.job_id} for target: {target}")


def start(cfg):
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
    loss_difference_checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.dpath,
        save_top_k=1,
        monitor="loss_difference",
        filename="loss_difference_{loss_difference:.5f}_{epoch}",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.dpath,
        save_top_k=1,
        every_n_epochs=1,
        filename="last_{epoch}",
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
            max_epochs=cfg.train.num_epochs,
            callbacks=[
                checkpoint_callback,
                early_stopping,
                last_checkpoint_callback,
                loss_difference_checkpoint_callback,
            ],
            enable_progress_bar=False,
            log_every_n_steps=1,
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
                last_checkpoint_callback,
                loss_difference_checkpoint_callback,
            ],
            log_every_n_steps=1,
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
        f"Completed training:\n TARGET = {cfg.target}\n DATASET = {cfg.train.spath}\n NUM EPOCHS = {cfg.train.num_epochs}\n TRAINING TIME = {(end - start) / 60:.3f} minutes"
    )

    write_results_yaml(
        cfg,
        data={
            "training_time": float((end - start) / 60),
            "train_loss": model.train_loss_plot,
            "train_acc": model.train_acc_plot,
            "val_loss": model.val_loss_plot,
            "val_acc": model.val_acc_plot,
        },
    )

    Utils.plot_loss_acc(
        values=model.train_loss_plot,
        dpath=Path(cfg.train.dpath).joinpath("train_loss.png"),
        type="Train Loss",
    )
    Utils.plot_loss_acc(
        values=model.train_acc_plot,
        dpath=Path(cfg.train.dpath).joinpath("train_acc.png"),
        type="Train Acc",
    )
    Utils.plot_loss_acc(
        values=model.val_loss_plot,
        dpath=Path(cfg.train.dpath).joinpath("val_loss.png"),
        type="Val Loss",
    )
    Utils.plot_loss_acc(
        values=model.val_acc_plot,
        dpath=Path(cfg.train.dpath).joinpath("val_acc.png"),
        type="Val Acc",
    )

    # === BEST CHECKPOINT ===#
    checkpoints = get_checkpoint_name(Path(cfg.train.dpath))

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
        dpath=Path(cfg.train.dpath).joinpath(f"{cfg.target}_prediction_results.csv"),
        target=cfg.target,
    )

    # === LAST CHECKPOINT ===#
    checkpoints = get_last_checkpoint_name(Path(cfg.train.dpath))

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
            Path(cfg.train.dpath).joinpath(
                f"last_{cfg.target}_prediction_results.yaml",
            )
        ),
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)

    Utils.plot_fit(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        dpath=Path(cfg.train.dpath).joinpath(f"last_{cfg.target}_fit.png"),
        target=cfg.target,
    )

    Utils.write_csv_results(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        names=model.sample_names,
        dpath=Path(cfg.train.dpath).joinpath(
            f"last_{cfg.target}_prediction_results.csv"
        ),
        target=cfg.target,
    )

    # === LOSS DIFFERENCE CHECKPOINT ===#
    checkpoints = get_difference_checkpoint_name(Path(cfg.train.dpath))

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
            Path(cfg.train.dpath).joinpath(
                f"loss_difference_{cfg.target}_prediction_results.yaml",
            )
        ),
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)

    Utils.plot_fit(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        dpath=Path(cfg.train.dpath).joinpath(f"loss_difference_{cfg.target}_fit.png"),
        target=cfg.target,
    )

    Utils.write_csv_results(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        names=model.sample_names,
        dpath=Path(cfg.train.dpath).joinpath(
            f"loss_difference_{cfg.target}_prediction_results.csv"
        ),
        target=cfg.target,
    )

    print(
        f"Prediction on target `{cfg.target}` completed ✅:\n🔺 Maximum % error \\= `{np.max(model.errors):.5f}%`\n🔸 Mean % error \\= `{np.mean(model.errors):.5f}%`\n🔹 STD % error \\= `{np.std(model.errors):.5f}%`"
    )


if __name__ == "__main__":
    main()
