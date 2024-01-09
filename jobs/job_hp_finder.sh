#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=4:00:00
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

srun python /home/tommaso/git_workspace/GrapheNet/hp_finder.py target=electron_affinity > hp_finder.output

srun python /home/tommaso/git_workspace/GrapheNet/hp_finder.py target=electronegativity > hp_finder.output

srun python /home/tommaso/git_workspace/GrapheNet/hp_finder.py target=Fermi_energy > hp_finder.output

srun python /home/tommaso/git_workspace/GrapheNet/hp_finder.py target=ionization_potential > hp_finder.output

# srun python /home/tommaso/git_workspace/GrapheNet/hp_finder.py target=formation_energy > hp_finder.output