#!/bin/bash
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
mkdir -p $CODE_SERVER_EXTENSIONS
PATH="$HOME/.local/bin:$PATH"
PROJECT_DIR="$PWD"

module purge
#module load rl9-gpustack
#module load machine_learning

# activate the conda base and set the cache directory to the /ibex/user/${USER}/conda_cache directory
source /ibex/user/${USER}/miniconda3/bin/activate 
export CONDA_PKGS_DIRS=/ibex/user/${USER}/conda_cache

# activate a conda environment
export ENV_PREFIX=/ibex/user/${USER}/miniconda3/envs/continue_ai
#export ENV_PREFIX=$PWD/env
conda activate $ENV_PREFIX

# setup for OLLAMA
# Define variables
OLLAMA_URL="https://ollama.com/download/ollama-linux-amd64.tgz"
OLLAMA_TAR="ollama-linux-amd64.tgz"
export OLLAMA_BINARY="$(pwd)/bin/ollama"

# Check if Ollama binary exists
if [ ! -f "$OLLAMA_BINARY" ]; then
    echo "Ollama binary not found. Downloading..."
    curl -L $OLLAMA_URL -o $OLLAMA_TAR

    echo "Extracting Ollama binary..."
    tar -xzf $OLLAMA_TAR

    echo "Cleaning up..."
    rm $OLLAMA_TAR
fi

# Set the Ollama models directory
export OLLAMA_MODELS=/ibex/user/${USER}/models
# Create the models directory if it does not exist
mkdir -p $OLLAMA_MODELS 


# Run Ollama in the background and do not create a log file
# nohup $OLLAMA_BINARY serve & > /dev/null 2>&1
# Run Ollama in the background and send the output to nohup.out
nohup $OLLAMA_BINARY serve & > nohup.out

# Pull models from Ollama if they are not already present in the models directory
if [ ! -f "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/qwen2.5/1.5b" ]; then
    $OLLAMA_BINARY pull qwen2.5:1.5b
fi

if [ ! -f "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/llama3.2/3b" ]; then
    $OLLAMA_BINARY pull llama3.2:3b
fi

# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
CODE_SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "
this is the port from SLURM ${SLURM_STEP_RESV_PORTS}
To connect to the compute node ${COMPUTE_NODE} on Ibex running your Code Server, 
you need to create an ssh tunnel from your local machine to login node on Ibex 
using the following command.

ssh -L localhost:${CODE_SERVER_PORT}:${COMPUTE_NODE}:${CODE_SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${CODE_SERVER_PORT}

" >&2

# launch code server
code-server --auth none --bind-addr ${COMPUTE_NODE}:${CODE_SERVER_PORT} --extensions-dir=${CODE_SERVER_EXTENSIONS} "$PROJECT_DIR"
