#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=G_coulomb
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pl

export PATH=/home/tommaso/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/tommaso/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#INSERT YOUR SCRIPT HERE

srun python /home/tommaso/git_workspace/GrapheNet/coulomb_train_lightning.py target=total_energy train.base_lr=0.1368859516416597 atom_types=1 train.spath=/home/tommaso/git_workspace/GrapheNet/Coulomb_G/training_dataset resolution=450 > test.output