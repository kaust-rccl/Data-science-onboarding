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

mkdir -p /local/scratch/$USER

echo "Copying data to local sotrage"
cp -r $PWD/cifar-10-batches-py /local/scratch/$USER
cd /local/scratch/$USER
echo "Where is the dataset?"
echo "PWD: $(pwd)"
tree

echo "Now checking if python script can access the data"
python ${SLURM_SUBMIT_DIR}/train.py


sleep 100
