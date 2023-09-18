try:

    from lib.lib_dataset_generator import DatasetGenerator
    from lib.lib_utils import Utils
    import time
    import hydra
    from pathlib import Path

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="dataset")
def main(cfg):

    start = time.time()

    # Utils.create_subset_xyz(
    #     xyz_path=Path()
    #     .resolve()
    #     .parent.joinpath("Chemception", "data", "xyz_files_opt"),
    #     dpath=Path(cfg.path_xyz),
    #     n_items=7000,
    #     targets=cfg.features,
    #     oxygen_distribution_threshold=cfg.randomly.oxygen_outliers_th,
    #     min_num_atoms=cfg.randomly.min_num_atoms,
    #     drop_custom=True,
    # )

    dataset = DatasetGenerator(cfg)

    end = time.time()

    print(f"\nDataset generated in = {(end-start)/60:.3f} minutes\n")


if __name__ == "__main__":
    main()
