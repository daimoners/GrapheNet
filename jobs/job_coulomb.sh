#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=Coulomb_Matrices
#SBATCH --error=coulomb.error
#SBATCH --output=coulomb.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pytorch

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > coulomb.output

srun python /home/tommaso/git_workspace/GrapheNet/coulomb_dataset_generator.py > coulomb.output