#!/bin/bash

#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=50G

module load machine_learning
export OMP_NUM_THREADS=1
srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} python -m torch.distributed.launch --nproc_per_node=${SLURM_GPUS_PER_NODE}  train_ddp.py 
 



