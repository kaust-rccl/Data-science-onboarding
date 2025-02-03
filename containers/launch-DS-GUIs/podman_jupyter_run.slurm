#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=32G
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

echo -e "

${node} pinned to port ${port}

You can now view your Jupyter Lab in your browser.

Network URL: http://${local_ip}:${port}
Network URL: http://${node}.kaust.edu.sa:${port}

" >&2

mkdir -p /run/user/${UID}/bus /ibex/user/${user}/podman_images


# launch podman

podman run \
-e NVIDIA_VISIBLE_DEVICES='' \
--rm \
-p ${port}:8888 \
-p 8501:8501 \
-v ${PWD}:/app/mycode \
--device=nvidia.com/gpu=all \
--security-opt=label=disable \
--root=/ibex/user/${user}/podman_images \
abdelghafour1/tf_pip_gpu_vf:tf_gpu \
jupyter lab --ip=0.0.0.0 --allow-root