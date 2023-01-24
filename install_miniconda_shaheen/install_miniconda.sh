# install miniconda in user's $HOME directory
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install miniconda in user's miniconda3 directory
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3

#bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh

# creat a conda_cache directory in user's $HOME directory
mkdir -p $PWD/conda_cache
export CONDA_PKGS_DIRS=$PWD/conda_cache

# activate conda base from the command line
source $PWD/miniconda3/bin/activate

# update conda to most recent version (if necessary)
conda update --name base --channel defaults --yes conda

# install mamba (faster, experimental package manager) from Conda Forge
conda install --name base --channel conda-forge mamba --yes

# make sure that base environment is not active by default
conda config --set auto_activate_base false