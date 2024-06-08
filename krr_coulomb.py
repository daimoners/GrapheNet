try:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.metrics import mean_squared_error
    import submitit
    import hydra
    from pathlib import Path
    from tqdm.rich import tqdm
    from omegaconf import OmegaConf, open_dict
    import pandas as pd
    import matplotlib.pyplot as plt
    from lib.lib_utils import Utils
    from lib.lib_coulomb import (
        sort_by_row_norm,
        standardize_matrix,
        padd_matrix,
        calculate_coulomb_matrix,
        read_xyz,
    )
    import xgboost as xgb
    import yaml
    import time

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_KRR:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        krr(self.args)


@hydra.main(version_base="1.2", config_path="config", config_name="krr")
def main(args):
    executor = submitit.AutoExecutor(
        folder=str(Path(args.slurm_output)),
        slurm_max_num_timeout=30,
    )

    executor.update_parameters(
        mem_gb=0 if not args.slurm_mem else args.slurm_mem,
        tasks_per_node=1,
        gpus_per_node=0 if not args.slurm_ngpus else args.slurm_ngpus,
        cpus_per_task=2 if not args.slurm_ncpus else args.slurm_ncpus,
        timeout_min=args.slurm_timeout,
        slurm_partition=args.slurm_partition,
        slurm_exclude=args.slurm_exclude,
    )

    if args.slurm_nodelist:
        executor.update_parameters(
            slurm_additional_parameters={"nodelist": f"{args.slurm_nodelist}"}
        )

    executor.update_parameters(name=f"{args.slurm_job_name}")
    slurm_auto_dftb = SLURM_KRR(args)
    job = executor.submit(slurm_auto_dftb)
    print(f"Submitted job_id: {job.job_id}")


def krr(args):

    # === Dataset Train === #
    train_df = pd.read_csv(Path(args.dataset_path).joinpath("train", "train.csv"))
    # train_df = train_df[: args.train_size]
    samples = [
        f.stem
        for f in Path(args.dataset_path).joinpath("train").iterdir()
        if f.suffix.lower() == ".npy"
    ]
    train_df = train_df.loc[train_df["file_name"].isin(samples)]
    train_values = train_df[f"{args.target}"].values
    train_matrices = np.empty((len(samples), args.resolution**2), dtype=np.ndarray)
    matrices_paths = [
        f
        for f in Path(args.dataset_path).joinpath("train").iterdir()
        if f.suffix.lower() == ".npy"
    ]
    pbar = tqdm(total=len(matrices_paths))
    for i, m in enumerate(matrices_paths):
        matrix = np.load(m)
        matrix = sort_by_row_norm(matrix)
        matrix = padd_matrix(matrix, args.resolution)
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        matrix = matrix.flatten()
        train_matrices[i] = matrix
        pbar.update(1)
        pbar.refresh()
    pbar.close()

    # === Dataset Test === #
    test_df = pd.read_csv(Path(args.dataset_path).joinpath("test", "test.csv"))
    # test_df = test_df[: args.test_size]
    samples = [
        f.stem
        for f in Path(args.dataset_path).joinpath("test").iterdir()
        if f.suffix.lower() == ".npy"
    ]
    test_df = test_df.loc[test_df["file_name"].isin(samples)]
    test_names = test_df["file_name"].to_list()
    test_values = test_df[f"{args.target}"].values
    test_matrices = np.empty((len(samples), args.resolution**2), dtype=np.ndarray)
    matrices_paths = [
        f
        for f in Path(args.dataset_path).joinpath("test").iterdir()
        if f.suffix.lower() == ".npy"
    ]
    pbar = tqdm(total=len(matrices_paths))
    for i, m in enumerate(matrices_paths):
        matrix = np.load(m)
        matrix = sort_by_row_norm(matrix)
        matrix = padd_matrix(matrix, args.resolution)
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
        matrix = matrix.flatten()
        test_matrices[i] = matrix
        pbar.update(1)
        pbar.refresh()
    pbar.close()

    # === Kernel Ridge Regression ===#
    # krr = KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    # krr.fit(train_matrices, train_values)
    # y_pred = krr.predict(test_matrices)

    # === Linear Regression ===#
    # model = LinearRegression()
    # model.fit(train_matrices, train_values)
    # y_pred = model.predict(test_matrices)

    # === XGBoost === #
    dtrain = xgb.DMatrix(train_matrices, label=train_values)
    dtest = xgb.DMatrix(test_matrices)

    # Configura i parametri per XGBoost
    params = {
        "objective": "reg:squarederror",  # Per la regressione
        "eval_metric": "rmse",  # Metrica di valutazione
        "eta": 0.1,  # Tasso di apprendimento
        "max_depth": 6,  # Profondità massima dell'albero
        "subsample": 0.8,  # Frazione di campioni da utilizzare per l'addestramento di ciascun albero
        "colsample_bytree": 0.8,  # Frazione di features da utilizzare per l'addestramento di ciascun albero
        "tree_method": "hist",
        "device": "cuda",
        "reg_alpha": 0.1,  # regolarizzazione L1
        "reg_lambda": 0.1,  # regolarizzazione L2
    }

    # Addestra il modello XGBoost
    num_rounds = 150  # Numero di iterazioni di boosting
    start = time.time()
    bst = xgb.train(params, dtrain, num_rounds)
    end = time.time()

    # Effettua le predizioni
    y_pred = bst.predict(dtest)

    Utils.plot_fit(
        y=test_values,
        y_hat=y_pred,
        dpath=Path(
            f"/home/tommaso/git_workspace/GrapheNet/krr/XGB_fit_{args.target}.png"
        ),
        target=args.target,
    )

    df = Utils.write_csv_results(
        y=test_values,
        y_hat=y_pred,
        names=test_names,
        dpath=Path(
            f"/home/tommaso/git_workspace/GrapheNet/krr/XGB_fit_{args.target}.csv"
        ),
        target=args.target,
    )

    performance = {
        "training_time": float((end - start) / 60),
        "Maximum % error": float(np.max(df[f"{args.target}_MAE"])),
        "Mean % error": float(np.mean(df[f"{args.target}_MAE"])),
        "STD % error": float(np.std(df[f"{args.target}_MAE"])),
    }

    with open(
        f"/home/tommaso/git_workspace/GrapheNet/krr/XGB_{args.target}_prediction_results.yaml",
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)


if __name__ == "__main__":
    # args = OmegaConf.create(
    #     {
    #         "dataset_path": "/home/tommaso/git_workspace/GrapheNet/data_paper/subset_Coulomb_GO/training_dataset",
    #         "target": "ionization_potential",
    #         "resolution": 1263,
    #     }
    # )
    # krr(args)
    main()
