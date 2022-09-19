#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00

module load machine_learning
module list

srun -u python device_info.py
