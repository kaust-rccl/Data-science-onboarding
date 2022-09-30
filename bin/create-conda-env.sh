#!/bin/bash --login
# entire script fails if a single command fails
set -e

# create the conda environment
export ENV_PREFIX=$PWD/env
mamba env create --prefix $ENV_PREFIX --file environment.yml --force
#mamba activate $ENV_PREFIX
#. postBuild
