#!/bin/bash --login
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=16G
#SBATCH --partition=batch 
#SBATCH --job-name=launch-code-server
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err

# setup the environment

export CODE_SERVER_CONFIG=~/.config/code-server/config.yaml
export XDG_CONFIG_HOME=$HOME/tmpdir
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env
PATH="$HOME/.local/bin:$PATH"

module purge
module load machine_learning
# conda activate "$ENV_PREFIX"

# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
CODE_SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "
echo this is the port from SLURM ${SLURM_STEP_RESV_PORTS}
To connect to the compute node ${COMPUTE_NODE} on Ibex running your Code Server, 
you need to create an ssh tunnel from your local machine to login node on Ibex 
using the following command.

ssh -L localhost:${CODE_SERVER_PORT}:${COMPUTE_NODE}:${CODE_SERVER_PORT} ${USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${CODE_SERVER_PORT}

" >&2

# launch code server
code-server --auth none --bind-addr ${COMPUTE_NODE}:${CODE_SERVER_PORT} "$PROJECT_DIR"