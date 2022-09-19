#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=2
#SBATCH --constraint=v100
#SBATCH --time=00:05:00

module load machine_learning
module list

srun -u -l python device_info.py
