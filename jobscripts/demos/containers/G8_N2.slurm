#!/bin/bash

#SBATCH --gpus=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=200G
#SBATCH --account=ibex-cs

module load singularity
export IMAGE=./pytorch_22.08-py3.sif
export BIND_MOUNT="-B /ibex/ai/home/shaima0d/KSL_Trainings"

export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_NCCL_SOCKET_IFNAME=ib0
export SINGULARITYENV_NCCL_DEBUG=INFO

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=10121
echo "head node is ${master_ip}:${master_port}"


for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
      singularity exec --nv ${BIND_MOUNT} ${IMAGE} \
      python -m torch.distributed.launch \
     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port}\
     train_ddp.py &
 
done
wait
