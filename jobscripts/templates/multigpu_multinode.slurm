#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=16
#SBATCH --gpus-per-node=4
#SBATCH --constraint=[v100]&[cpu_intel_platinum_8260]
#SBATCH --time=00:05:00

#module load machine_learning/2022.11
#module list

#module load openmpi
#module load cudnn/8.1.1-cuda11.2.2
#module load gcc/9.2.0

source ~/miniconda3/bin/activate /home/barradd/miniconda3/envs/ml-env

srun -u -l python device_info.py
