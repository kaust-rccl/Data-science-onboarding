#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=workq
#SBATCH --time=00:30:00 
#SBATCH -A k1033
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

### Load the modules you need for your job
### and swap the PrgEnv module if needed
module swap PrgEnv-cray PrgEnv-intel
module load intelpython3/2022.0.2.155 pytorch/1.8.0

############################################################
## Load the conda base and activate the conda environment ##
############################################################
############################################################
## export the path to the conda base and conda_cache
############################################################
# export ENV_PREFIX=/project/k1033/barradd/install_miniconda_shaheen
# export CONDA_PKGS_DIRS=$PWD/conda_cache
############################################################ 
## activate conda base from the command line
############################################################
#source $ENV_PREFIX/miniconda3/bin/activate $ENV_PREFIX/env
## Or be very explicit and use the full path to the activate script
#source /project/k1033/barradd/install_miniconda_shaheen/miniconda3/bin/activate /project/k1033/barradd/install_miniconda_shaheen/env


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
To connect to the compute node ${node} on Shaheen running your jupyter notebook server,
you need to run following two commands in a terminal
1. Command to create ssh tunnel from you workstation/laptop to cdlX:
ssh -L ${port}:localhost:${port} ${user}@${submit_host}.hpc.kaust.edu.sa
2. Command to create ssh tunnel to run on cdlX:
ssh -L ${port}:${node}:${port} ${user}@${gateway}

Copy the link provided below by jupyter-server and replace the nid0XXXX with localhost before pasting it in your browser on your workstation/laptop. Do not forget to close the notebooks you open in you browser and shutdown the jupyter client in your browser for gracefully exiting this job or else you will have to mannually cancel this job running your jupyter server.
"

echo "Starting jupyter server in background with requested resouce"

# Run Jupyter
# jupyter lab --no-browser --port=${port} --ip=${node} 
jupyter ${1:-lab} --no-browser --port=${port} --port-retries=0  --ip=${node}
