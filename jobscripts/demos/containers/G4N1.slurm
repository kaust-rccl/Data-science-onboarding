#!/bin/bash

#SBATCH --job-name=torch_ngc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
#SBATCH --time=00:05:00

module load singularity


export IMAGE=./pytorch_22.08-py3.sif
export BIND_MOUNT="-B /ibex/ai/home/shaima0d/KSL_Trainings"
export SINGULARITYENV_OMP_NUM_THREADS=1

srun -n 1 -c ${SLURM_CPUS_PER_TASK} singularity exec --nv ${BIND_MOUNT} $IMAGE python -m torch.distributed.launch --nproc_per_node=4 ./train_ddp.py


