#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=predict_GrapheNet
#SBATCH --error=predict.error
#SBATCH --output=predict.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pytorch

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > predict.output

srun python /home/tommaso/git_workspace/GrapheNet/predict_lightning.py > predict.output