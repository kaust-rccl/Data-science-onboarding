#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=16
#SBATCH --gpus-per-node=4
#SBATCH --constraint=[v100]&[cpu_intel_platinum_8260]
#SBATCH --time=00:05:00

module load machine_learning/2022.11
#module list

srun -u -l python device_info.py
