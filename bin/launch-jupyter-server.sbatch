#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=16G
#SBATCH --partition=batch 
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=out/%x-%j-slurm.out
#SBATCH --error=err/%x-%j-slurm.err

# use srun to launch Jupyter server in order to reserve a port
srun --resv-ports=1 ./bin/launch-jupyter-server.srun
