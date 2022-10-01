#!/bin/bash 
#SBATCH --job-name=G1N1B256
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=400G
#SBATCH --output=%x-%J.out
#SBATCH --error=%x-%J.out

module load cuda/11.2.2
module load dl
module load pytorch/1.9.0
module load torchvision
module load horovod/0.22.1_torch
module list

echo "Hostnames: $SLURM_NODELIST"

export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=tcp

export RUNDIR=${PWD}/result_${SLURM_JOB_NAME}_${SLURM_JOBID}
mkdir -p $RUNDIR

# ImageNet dataset 1000 classes
  ## local storage
if [ -z "${DATA_DIR}" ]; then
  export DATA_DIR="/local/reference/CV/ILSVR/classification-localization/data/jpeg"
fi

batch_size=256
epochs=5
workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "Data source: $DATA_DIR"
echo "Using Batch size : $batch_size"
echo "Epochs : $epochs"
echo "CPU workers: $workers"

cd $RUNDIR
main_exe="../../../scripts/train_resnet50.py"
cmd="python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"
echo "time -p srun -u -n ${SLURM_NTASKS}  -c ${SLURM_CPUS_PER_TASK} ${cmd} --log-dir=log.${SLURM_JOBID} --warmup-epochs=0.0"
time -p srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} -c ${SLURM_CPUS_PER_TASK} ${cmd} --log-dir=log.${SLURM_JOBID} --warmup-epochs=0.0 &> output.txt
    
