#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=ibex-cs

source ~/miniconda3/bin/activate ray_demo
#Requested number of workers
if [ -z ${NUM_WORKERS} ] ; then
  NUM_WORKERS=1
else
  NUM_WORKERS=${NUM_WORKERS}
fi


echo "
Connect to dashboard by creating SSH tunnels. Copy the following command in a new terminal
and connect to localhost via your browser.
ssh -L 9122:localhost:9122 -L 9123:localhost:9123 ${USER}@${HOSTNAME}.ibex.kaust.edu.sa
"

export server_port=9121
export dashboard_port=9122
export tensorboard_port=9123
export TB_TMPDIR=$PWD/tboard/${SLURM_JOBID}
mkdir -p ${TB_TMPDIR}

export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -I | cut -d " " -f 2)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password}" > head_node_info


ray start --node-ip-address ${head_node_ip} --port ${server_port} --redis-password=${redis_password} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=127.0.0.1 \
        --num-cpus 1 --block &
tensorboard --logdir=${PWD}/logs/${SLURM_JOBID} --port=${tensorboard_port} & 
sleep 20

job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ ))
 do
   job_ids[$i]=$(sbatch -x $SLURM_NODELIST worker_node.slurm | cut -d " " -f 4)
 done 

while [ ! -z $(squeue -n ray_worker -t PD -h -o %A) ]
do
	echo "Waiting for worker(s) to start"
        sleep 20
done


python -u ray_mnist_pytorch.py --use-gpu \
          --cpus-per-trial=4 --gpus-per-trial=1 \
          --num-samples=50 \
          --max-concurrent-trials=32 
#          --resume --logs-dir=/ibex/scratch/shaima0d/impact_project_2020/ray_cluster/KSL_hpo_tutorial/demo2/logs/21428634
          


# Shutdown workers before the head node
touch $PWD/shutdown.txt
sleep 20
echo " Stopping ray on Head node: $(/bin/hostname)"
ray stop
rm $PWD/shutdown.txt

 
