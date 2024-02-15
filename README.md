# GrapheNet

This is the code used to produce the results presented in the paper called **GrapheNet: a deep learning framework for predicting the physical and electronic properties of nanographenes using images**.  
The paper is about using computer vision techniques (and in particular convolutional neural networks) to predict the electronic properties of graphene and graphene-oxyde flakes. The neural networks are implemented in `Python` using the `PyTorch` framework. Also the framewrok take care to create the dataset starting from a folder of **.xyz** files and a **.csv** file containing the target properties.

## Usage

## Requirements

* CUDA
* miniconda3

### Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/daimoners/GrapheNet.git
   ```

2. Create the conda env from the conda_env.yaml file:

   ```bash
   conda env create -f ./GrapheNet/conda_env.yaml
   ```

3. Activate the conda env and enter the GrapheNet directory:

   ```bash
   conda activate pl && cd GrapheNet
   ```

### Configuration

1. Customize the **dataset.yaml** file in the **config** folder according to your needs:

   * **data_folder**: specify the name of the data folder.
   * **dataset_name**: specify the name of the dataset.
   * **dataset_csv_path**: specify the path of the .csv file containing the target properties along with the names of the single .xyz files.
   * **features**: specity a list of properties of interest, in order to drop the unused columns of the .csv file defined in **dataset_csv_path**
   * **plot_distributions**: a flag that determines whether to plot the distributions of target properties for train/val/test .
   * **augmented_png**: a flag that determines whether to augment the images by rotating them by 90, 180, 270 degrees. (i.e. the size of the dataset will be quadrupled).
   * **augmented_xyz**: a flag that determines whether to augment the images by rotating them by a 3 randoms angle within [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]. (i.e. the size of the dataset will be quadrupled).


2. Customize your training configurations using Hydra. Configuration files are located in the `config` directory. Modify the existing files or create new ones according to your needs.

   * **name**: name of the current hydra config file.
   * **atom_types**: specify the different type of atoms in the dataset (i.e. 3 for GO and 1 for DG).
   * **cluster**: a flag that determines whether the training should be performed on cluster or not.
   * **coulomb**: a flag that determines whether the dataset is composed by coulomb matrices rather than png images.
   * **deterministic**: a flag that set the experiment to be reproducible.
   * **num_workers**: set the num workers for the Pytorch Dataloaders.
   * **enlargement_method**: you can choose between **padding** or **resize** and specify the enlargement method in order to have the same size for all the images (work only with images dataset).
   * **resolution**: sets the maximum size to which the images/coulomb matrices of the dataset will be resized.
   * **target**: sets the target property on which the network is trained.
   * **train**:
      ```
      base_lr: set the initial learning rate.
      batch_size: set the batch size
      compile: a flag that set the model to be compiled or not.
      lr_list: it contains a list of optimized learning rate for the datasets used in this paper.
      matmul_precision: set the pytorch matmul precision (high or medium).
      num_epochs: set the number of epochs.
      dataset_path: sets the dataset on which the network will be trained.
      ```

### Training

 Launch the **train_lightning.py** with the appropriate hydra config file:

   ```bash
   python train_lightning.py -cn train_predict.yaml
   ```

### Evaluation

 The evaluation is automatically performed at the end of the training. If you want to perform again the evaluation on the test set, launch the **predict_lightning.py** with the appropriate hydra config file:

   ```bash
   python predict_lightning.py -cn train_predict.yaml
   ```

### Automatic learning rate optimization

In order to optimize the learning rate for each target, the **hp_finder.py** script can be launched with the appropriate hydra config file:
   
   ```bash
   python hp_finder.py -cn train_predict.yaml
   ```
Alternatively, the **automatic_opt_and_training.py** script take care to optimize the learning rate for the target, populate the **train.lr_list** with the optimized learning rate and then training the model with the optimized learning rate. The operation is performed for each target key in the **train.lr_list**, only if the correspondent learning rate is missing or equal to zero. Otherwise the model is trained on each target with the learning rates stored in **train.lr_list**.
   ```bash
   python automatic_opt_and_training.py -cn train_predict.yaml
   ```
