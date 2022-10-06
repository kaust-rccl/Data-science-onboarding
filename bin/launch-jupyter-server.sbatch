#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8  
#SBATCH --mem=32G
#SBATCH --partition=batch 
#SBATCH --job-name=demo
#SBATCH --reservation=HACKATHON
#SBATCH --mail-type=ALL
#SBATCH --output=out/%x-%j-slurm.out
#SBATCH --error=err/%x-%j-slurm.err

# use srun to launch Jupyter server in order to reserve a port
srun --resv-ports=1 ./bin/launch-jupyter-server.srun
