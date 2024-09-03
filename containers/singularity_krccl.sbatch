#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --time=00:15:00 
#SBATCH --partition=debug 
#SBATCH --mem=16G
#SBATCH --job-name=build
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err

module load singularity
mkdir -p /ibex/user/$USER/singularity_cache /ibex/user/$USER/TMPDIR
export SINGULARITY_TMPDIR=/ibex/user/$USER/TMPDIR
export SINGULARITY_CACHEDIR=/ibex/user/$USER/singularity_cache
unset SINGULARITY_BIND

#singularity pull docker://krccl/horovod_gpu:0192
singularity build -f --nv --force $PWD/horovod_krccl.sif $PWD/horovod_krccl.def
