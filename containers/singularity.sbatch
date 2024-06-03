#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --time=00:15:00 
#SBATCH --partition=batch 
#SBATCH --mem=32G
#SBATCH --job-name=build
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err

module load singularity
mkdir -p /ibex/user/$USER/singularity_cache
export SINGULARITY_CACHEDIR=/ibex/user/$USER/singularity_cache
unset SINGULARITY_BIND

singularity build -f --nv --force $PWD/horovod.sif $PWD/horovod.def
