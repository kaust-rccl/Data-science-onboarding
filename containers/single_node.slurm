#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=128G
#SBATCH --partition=batch 
#SBATCH --job-name=horovod_demo
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 


module load openmpi/4.1.4/gnu11.2.1-cuda11.8
module load singularity

export IMAGE=$PWD/horovod.sif

echo "PyTorch with Horovod"
mpirun -np 4  singularity exec --nv $IMAGE python ./pytorch_synthetic_benchmark.py --model resnet50 --batch-size 128 --num-warmup-batches 10 --num-batches-per-iter 10 --num-iters 10 >>pytorch_1node.log

echo "Tensorflow2 with Horovod"
mpirun -np 4  singularity exec --nv $IMAGE python ./tensorflow2_synthetic_benchmark.py --model ResNet50  --batch-size 128 --num-warmup-batches 10 --num-batches-per-iter 10 --num-iters 10 >> TF2_1node.log

