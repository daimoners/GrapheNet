#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=train_GrapheNet_coulomb
#SBATCH --error=train_coulomb_2.error
#SBATCH --output=train_coulomb_2.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pl

export PATH=/home/tommaso/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/tommaso/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > train_coulomb_2.output

srun python /home/tommaso/git_workspace/GrapheNet/coulomb_train_lightning.py target=electron_affinity train.base_lr=0.061813296683228255 > train_coulomb_2.output

# srun python /home/tommaso/git_workspace/GrapheNet/coulomb_predict_lightning.py > train_coulomb.output