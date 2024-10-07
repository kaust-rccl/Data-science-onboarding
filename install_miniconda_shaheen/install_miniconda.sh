# create the folder my software and go there
mkdir -p $MY_SW && cd $MY_SW

# execute the local miniconda for shaheen
bash /sw/sources/miniconda/conda24.1.2-python3.12.1/Miniconda3-latest-Linux-x86_64.sh -b -s -p $MY_SW/miniconda3-amd64 -u

# activate conda base from the command line
# you might need this on your scripts
source $MY_SW/miniconda3-amd64/bin/activate

# install mamba (faster, Vexperimental package manager) from Conda Forge
conda install -y -c conda-forge mamba

# exit conda
conda deactivate
