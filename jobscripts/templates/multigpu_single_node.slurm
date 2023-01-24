#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=v100
#SBATCH --time=00:05:00

module load machine_learning/2022.11
#module list

#module load openmpi
#module load cudnn/8.1.1-cuda11.2.2
#module load gcc/9.2.0

#source ~/miniconda3/bin/activate /ibex/scratch/barradd/ibex-machine-learning-modules/env 

srun -u python device_info.py
