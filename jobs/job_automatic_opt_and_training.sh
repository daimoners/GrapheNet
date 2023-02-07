#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=opt_train_GrapheNet
#SBATCH --error=opt_train.error
#SBATCH --output=opt_train.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pytorch

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > opt_train.output

srun python /home/tommaso/git_workspace/GrapheNet/automatic_opt_and_training.py > opt_train.output