#!/bin/bash

#SBATCH --job-name=my_first_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00

module load machine_learning
module list

echo "Howdy from $(/bin/hostname)"

echo "Copying data in Python script"
srun -n 1 python ${SLURM_SUBMIT_DIR}/inscript.py

