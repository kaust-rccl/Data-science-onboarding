#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 00:10:00 

module load singularity
mkdir -p /ibex/user/$USER/singularity_cache
export SINGULARITY_CACHEDIR=/ibex/user/$USER/singularity_cache

singularity build -f --force $PWD/horovod.sif $PWD/horovod.def
