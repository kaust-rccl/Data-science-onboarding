#!/bin/bash --login
# entire script fails if a single command fails
set -e
# define the folder to install 
export PREFIX=$HOME/miniconda3
# or define a path to install
# export PREFIX=/ibex/scratch/$USER/miniconda3

# download and install miniforge
# this will create the folder $PREFIX and install miniforge there
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh -b -p $PREFIX

# once installed, we need to activate the base environment
source $PREFIX/bin/activate

# update conda to most recent version (if necessary)
# conda update --name base --channel defaults --yes conda

# make sure that base environment is not active by default
conda config --set auto_activate_base false

# remove the installer
rm Mambaforge-$(uname)-$(uname -m).sh