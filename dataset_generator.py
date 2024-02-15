try:

    from lib.lib_dataset_generator import DatasetGenerator
    import time
    import hydra

except Exception as e:

    print("Some module are missing {}".format(e))


@hydra.main(version_base="1.2", config_path="config", config_name="dataset")
def main(cfg):

    start = time.time()

    dataset = DatasetGenerator(cfg)

    end = time.time()

    print(f"\nDataset generated in = {(end-start)/60:.3f} minutes\n")


if __name__ == "__main__":
    main()
