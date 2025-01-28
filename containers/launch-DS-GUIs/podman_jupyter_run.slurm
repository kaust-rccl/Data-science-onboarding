#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8  
#SBATCH --mem=16G
#SBATCH --partition=batch 
#SBATCH --job-name=jupyter
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err
# setup the environment
export XDG_RUNTIME_DIR=/tmp node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
local_ip=$(hostname -I  | awk -F" " '{print $1}')

# print tunneling instructions  
echo -e " 
${node} pinned to port ${port} on ${submit_host} 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 
 
ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} ${user}@glogin.ibex.kaust.edu.sa 
 
Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop.
" >&2 

mkdir -p /run/user/${UID}/bus 

# launch podman
# Use the full image name to avoid short-name resolution issues
IMAGE="docker.io/library/jupyter/datascience-notebook"

podman run \
-e NVIDIA_VISIBLE_DEVICES='' \
--rm \
-p ${port}:8888 \
-p 8501:8501 \
-v ${PWD}:/app/mycode \
--device=nvidia.com/gpu=all \
--security-opt=label=disable \
--root=/ibex/user/${user}/podman \
${IMAGE} \
jupyter lab --ip=0.0.0.0 --allow-root 
