#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --job-name=hp_find_GrapheNet
#SBATCH --error=hp_finder.error
#SBATCH --output=hp_finder.log
#
#---------------------------------------------------------------------------------------

module load slurm

eval "$(conda shell.bash hook)"

conda activate pl

export PATH=/home/tommaso/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/tommaso/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#INSERT YOUR SCRIPT HERE

echo $SLURM_JOB_ID > hp_finder.output

srun python /home/tommaso/git_workspace/GrapheNet/other_code/hp_finder.py