#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=2:30:00
#SBATCH --job-name=kfold_GrapheNet
#SBATCH --error=kfold.error
#SBATCH --output=kfold.log
#SBATCH --exclude=gn05,gn06,gn07
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pl

export PATH=/home/tommaso/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/tommaso/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > train.output

srun python /home/tommaso/git_workspace/GrapheNet/kfold_train_lightning.py > kfold.output
