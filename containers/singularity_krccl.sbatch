#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 01:00:00 

module load singularity
mkdir -p /ibex/user/$USER/singularity_cache /ibex/user/$USER/TMPDIR
export SINGULARITY_TMPDIR=/ibex/user/$USER/TMPDIR
export SINGULARITY_CACHEDIR=/ibex/user/$USER/singularity_cache
#singularity pull docker://krccl/horovod_gpu:0192
singularity build -f --force $PWD/horovod_krccl.sif $PWD/horovod_krccl.def
