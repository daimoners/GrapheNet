#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=predict_GrapheNet
#SBATCH --error=predict.error
#SBATCH --output=predict.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pl

export PATH=/home/tommaso/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/tommaso/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > predict.output

srun python /home/tommaso/git_workspace/GrapheNet/predict_lightning.py > predict.output