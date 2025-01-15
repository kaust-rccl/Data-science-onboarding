#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --partition=ppn
#SBATCH -A v10004
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 

##########################################################
# remember you have to execute this script from your     #
# /scratch/$USER  folder                                 #
##########################################################

############################################################
## Load the conda base and activate the conda environment ##
############################################################
############################################################
## export the path to the conda base and conda_cache       #
############################################################
export ENV_PREFIX=${MY_SW}/envs/continue_ai
#export ENV_PREFIX=${MY_SW}/envs/continue_ai_py3.9
#export ENV_PREFIX=${MY_SW}/envs/code-server
export CONDA_PKGS_DIRS=${MY_SW}/cache
############################################################ 
## activate conda base from the command line               #
############################################################
source ${MY_SW}/miniconda3-amd64/bin/activate ${ENV_PREFIX}

####################################################################
# Or be very explicit and use the full path to the activate script #
# like in the line below                                           #
####################################################################
#source /project/k1033/barradd/install_miniconda_shaheen/miniconda3/bin/activate /project/k1033/barradd/install_miniconda_shaheen/env

# setup the environment
export CODE_SERVER_CONFIG=${SCRATCH_IOPS}/config
export XDG_CONFIG_HOME=${SCRATCH}/.cache
export EXTENSIONS_DIR=${SCRATCH_IOPS}/.code/extensions
export CODE_DATADIR=${SCRATCH_IOPS}/.code/data
mkdir -p ${EXTENSIONS_DIR} ${CODE_DATADIR}

node=$(/bin/hostname -s)
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}

if [ -f "${CODE_SERVER_CONFIG}" ] ; then
rm ${CODE_SERVER_CONFIG}
fi

echo "bind-addr: ${node}:${port}" >> ${CODE_SERVER_CONFIG}
echo "auth: password" >> ${CODE_SERVER_CONFIG}
echo "password: 10DowningStreet" >> ${CODE_SERVER_CONFIG}
echo "cert: false" >> ${CODE_SERVER_CONFIG}

echo "
To connect to the compute node ${node} on Shaheen running your Code Server.
Copy the following two lines in a new terminal one after another to create a secure SSH tunnel between your computer and Shaheen compute node.

ssh -L ${port}:${node}:${port} ${USER}@${submit_host}.hpc.kaust.edu.sa 

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${port}

" >&2


code-server --auth=password --user-data-dir=${CODE_DATADIR} --extensions-dir=${EXTENSIONS_DIR} --verbose
