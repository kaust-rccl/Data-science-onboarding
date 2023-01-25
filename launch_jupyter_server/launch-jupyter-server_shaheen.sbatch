#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=workq
#SBATCH --time=00:30:00 
#SBATCH -A k1033
#SBATCH --error=dask_err_%j.err
#SBATCH --output=dask_out_%j.out

module swap PrgEnv-cray PrgEnv-intel

# export the path to the conda base and conda_cache
export ENV_PREFIX=/project/k1033/barradd/install_miniconda_shaheen
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CONDA_PKGS_DIRS=$PWD/conda_cache
 
 # activate conda base from the command line
 
source $ENV_PREFIX/miniconda3/bin/activate $ENV_PREFIX/env

# Or be very explicit and use the full path to the activate script
#source /project/k1033/barradd/install_miniconda_shaheen/miniconda3/bin/activate /project/k1033/barradd/install_miniconda_shaheen/env

mkdir workers${SLURM_JOBID}

# get tunneling info
export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
gateway=${EPROXY_LOGIN}
submit_host=${SLURM_SUBMIT_HOST}
port=8889
dask_dashboard=10001

echo $node on $gateway pinned to port $port
# print tunneling instructions jupyter-log
echo -e "
To connect to the compute node ${node} on Shaheen running your jupyter notebook server,
you need to run following two commands in a terminal
1. Command to create ssh tunnel from you workstation/laptop to cdlX:
ssh -L ${port}:localhost:${port} -L ${dask_dashboard}:localhost:${dask_dashboard} ${user}@${submit_host}.hpc.kaust.edu.sa
2. Command to create ssh tunnel to run on cdlX:
ssh -L ${port}:${node}:${port} -L ${dask_dashboard}:${node}:${dask_dashboard} ${user}@${gateway}

Copy the link provided below by jupyter-server and replace the nid0XXXX with localhost before pasting it in your browser on your workstation/laptop. Do not forget to close the notebooks you open in you browser and shutdown the jupyter client in your browser for gracefully exiting this job or else you will have to mannually cancel this job running your jupyter server.
"

echo "Starting dask server in background with requested resouce"

# Run Jupyter
jupyter lab --no-browser --port=${port} --ip=${node} 