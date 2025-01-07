#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:10:00
#SBATCH --partition=ppn
#SBATCH -A v10004
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 


##########################################################
# remember you have to execute this script from your     #
# /scratch/$USER  folder                                 #
##########################################################
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

##########################################################
#  Load the modules you need for your job               #
#  and swap the PrgEnv module if needed                 #
##########################################################
module swap PrgEnv-cray PrgEnv-intel
# module load intelpython3/2022.0.2.155 pytorch/1.8.0

############################################################
## Load the conda base and activate the conda environment ##
############################################################
############################################################
## export the path to the conda base and conda_cache       #
############################################################
export ENV_PREFIX=${MY_SW}/envs/jupyter
export CONDA_PKGS_DIRS=${MY_SW}/cache
############################################################ 
## activate conda base from the command line               #
############################################################
# source $ENV_PREFIX/miniconda3/bin/activate $ENV_PREFIX/install_miniconda_shaheen/env
source ${MY_SW}/miniconda3-amd64/bin/activate ${ENV_PREFIX}


####################################################################
# Or be very explicit and use the full path to the activate script #
# like in the line below                                           #
####################################################################
#source /project/k1033/barradd/install_miniconda_shaheen/miniconda3/bin/activate /project/k1033/barradd/install_miniconda_shaheen/env

# setup the environment
### Load the modules you need for your job

#module load python


export JUPYTER_CONFIG_DIR=${SCRATCH_IOPS}/.jupyter
export JUPYTER_DATA_DIR=${SCRATCH_IOPS}/.local/share/jupyter
export JUPYTER_RUNTIME_DIR=${SCRATCH_IOPS}/.local/share/jupyter/runtime
export IPYTHONDIR=${SCRATCH_IOPS}/.ipython

# setup ssh tunneling
# get tunneling info 
export XDG_RUNTIME_DIR=/tmp node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
gateway=${EPROXY_LOGIN}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo ${node} pinned to port ${port} on ${gateway} 

# print tunneling instructions jupyter-log
echo -e "
To connect to the compute node ${node} on Shaheen III running your jupyter notebook server,
you need to run following command in a new terminal on you workstation/laptop

ssh -L ${port}:${node}:${port} ${user}@${submit_host}.hpc.kaust.edu.sa

Copy the URL provided below by jupyter-server (one starting with http://127.0.0.1/...) and paste it in your browser on your workstation/laptop.

Do not forget to close the notebooks you open in you browser and shutdown the jupyter client in your browser for gracefully exiting this job or else you will have to manually cancel this job running your jupyter server.
"

echo "Starting jupyter server in background with requested resouce"

# Run Jupyter
# jupyter lab --no-browser --port=${port} --ip=${node} 
jupyter ${1:-lab} --no-browser --port=${port} --port-retries=0  --ip=${node}

