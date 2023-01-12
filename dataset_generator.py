try:

    from lib.lib_dataset_generator import DatasetGenerator
    from lib.lib_energy_components import generate_num_atoms
    import time
    import hydra
    from pathlib import Path

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(config_path="config", config_name="dataset")
def main(cfg):

    start = time.time()

    dataset = DatasetGenerator(cfg)

    end = time.time()

    if cfg.energy_components:
        generate_num_atoms(dataset_path=Path(cfg.dpath), xyz_path=Path(cfg.path_xyz))

    print(f"\nDataset generated in = {(end-start)/60:.3f} minutes\n")


if __name__ == "__main__":
    main()
