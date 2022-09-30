#!/bin/bash --login
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --job-name=remove-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# create the conda environment
export ENV_PREFIX="$PWD"/env
mamba remove -p ${ENV_PREFIX} --all > /dev/null
