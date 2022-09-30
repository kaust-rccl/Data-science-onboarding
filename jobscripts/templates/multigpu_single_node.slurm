#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
#SBATCH --time=00:05:00

module load machine_learning
module list

srun -u python device_info.py
