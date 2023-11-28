#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --job-name=create-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err

# entire script fails if a single command fails
set -e

# create the conda environment
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force
