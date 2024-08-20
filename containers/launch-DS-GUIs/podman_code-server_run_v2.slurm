#!/bin/bash --login
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=16G
#SBATCH --partition=batch 
#SBATCH --job-name=code-server
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err

# setup the environment
export CODE_SERVER_CONFIG=~/.config/code-server/config.yaml
export XDG_CONFIG_HOME=$HOME/tmpdir
export CODE_SERVER_EXTENSIONS=/ibex/user/$USER/code-server/extensions
export CODE_SERVER_local=/ibex/user/$USER/code-server/.local
export CODE_SERVER_config=/ibex/user/$USER/code-server/.config
mkdir -p $CODE_SERVER_EXTENSIONS $CODE_SERVER_local $CODE_SERVER_config
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env
PATH="$HOME/.local/bin:$PATH"

# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
CODE_SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
local_ip=$(hostname -I  | awk -F" " '{print $1}')

echo "
To connect to the compute node ${COMPUTE_NODE} on Ibex running your Code Server, 
you need to create an ssh tunnel from your local machine to login node on Ibex 
using the following command.

ssh -L localhost:${CODE_SERVER_PORT}:${COMPUTE_NODE}:${CODE_SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${CODE_SERVER_PORT}

" >&2

mkdir -p /run/user/${UID}/bus 

# launch podman container
echo ""

#podman run -d --name code-server -p 127.0.0.1:8080:8080 -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --root=/ibex/user/$USER/podman_images -e "DOCKER_USER=$USER" -e PUID="$(id -u)" -e PGID="$(id -g)"  -v "/ibex/user/$USER/Data-science-onboarding/:/home/" -v "/home/$USER/.config:/home/coder/.config" codercom/code-server
#podman run -d --name code-server -p 127.0.0.1:8080:8080 -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --root=/ibex/user/$USER/podman_images -e "DOCKER_USER=$USER" -e PUID="$(id -u)" -e PGID="$(id -g)"  -v "$CODE_SERVER_config:/home/.config"  codercom/code-server
#podman run -d --name code-server -p 127.0.0.1:8080:8080 -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --root=/ibex/user/$USER/podman_images -e "DOCKER_USER=$USER" -e PUID="$(id -u)" -e PGID="$(id -g)" -v "/home/$USER/.config:/home/coder/.config" codercom/code-server
#podman run -d --name code-server -p ${local_ip}:${CODE_SERVER_PORT}:8080 -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --root=/ibex/user/$USER/podman_images -e "DOCKER_USER=$USER" -e PUID="$(id -u)" -e PGID="$(id -g)"  -v "/ibex/user/$USER/Data-science-onboarding/:/home/" -v "$HOME/.config:/home/coder/.config" codercom/code-server
#podman run -d --name code-server -p ${CODE_SERVER_PORT}:8080 -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --root=/ibex/user/$USER/podman_images -e "DOCKER_USER=$USER" -e PUID="$(id -u)" -e PGID="$(id -g)"  -v "/ibex/user/$USER/Data-science-onboarding/:/home/" codercom/code-server


podman run -d -e NVIDIA_VISIBLE_DEVICES='' --device=nvidia.com/gpu=all --security-opt=label=disable --name code-server -e PUID="$(id -u)" -e PGID="$(id -g)" -p $local_ip:$CODE_SERVER_PORT:8443 -v /ibex/user/barradd/code-server/.config:/config --root=/ibex/user/$USER/podman_images -v /ibex/user/barradd/Data-science-onboarding:/app -v /ibex/user/barradd/:/defaults lscr.io/linuxserver/code-server:latest

podman exec -it --root=/ibex/user/$USER/podman_images code-server code-server --auth none --bind-addr ${COMPUTE_NODE}:${CODE_SERVER_PORT} --extensions-dir=${CODE_SERVER_EXTENSIONS} "$PROJECT_DIR" 

podman --root=/ibex/user/$USER/podman_images stop code-server
podman --root=/ibex/user/$USER/podman_images container rm code-server


