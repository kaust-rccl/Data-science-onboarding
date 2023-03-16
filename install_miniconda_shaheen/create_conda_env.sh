#!/bin/bash
# entire script fails if a single command fails
set -e
# export the environment prefix 
export ENV_PREFIX=$PWD/env
export CONDA_PKGS_DIRS=$PWD/conda_cache

# activate conda base from the command line
source $PWD/miniconda3/bin/activate 

# create the conda environment
mamba env create --prefix $ENV_PREFIX --file environment.yml --force