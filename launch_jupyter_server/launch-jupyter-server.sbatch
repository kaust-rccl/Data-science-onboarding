#!/bin/bash --login
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=16G
#SBATCH --partition=batch 
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 


# Load environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# setup the environment
module purge

# You can use the machine learning module 
module load machine_learning
# or you can activate the conda environment directly by uncommenting the following lines
#export ENV_PREFIX=$PWD/env
#conda activate $ENV_PREFIX

# setup ssh tunneling
# get tunneling info 
export XDG_RUNTIME_DIR=/tmp node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo ${node} pinned to port ${port} on ${submit_host} 

# print tunneling instructions  
echo -e " 
${node} pinned to port ${port} on ${submit_host} 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 
 
ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} ${user}@glogin.ibex.kaust.edu.sa 
 
Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop.
" >&2 
 
# Run Jupyter 
#jupyter notebook --no-browser --port=${port} --port-retries=0 --ip=${node}

# launch jupyter server
jupyter ${1:-lab} --no-browser --port=${port} --port-retries=0  --ip=${node}.ibex.kaust.edu.sa

