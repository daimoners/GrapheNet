#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=graphene
#SBATCH --error=job.error
#SBATCH --output=job.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pytorch

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > job.output

srun python /home/tommaso/git_workspace/Graphene/train_lightning.py > job.output