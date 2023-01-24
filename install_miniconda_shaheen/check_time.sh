#!/bin/bash
# entire script fails if a single command fails
set -e

export CONDA_PKGS_DIRS=$PWD/conda_cache
# activate conda base from the command line
source $PWD/miniconda3/bin/activate

# use time to measure the time taken by conda
mamba clean --all -y
time  mamba create -p $PWD/env -c conda-forge pandas python scikit-learn -d
echo "conda create time: $SECONDS seconds"
