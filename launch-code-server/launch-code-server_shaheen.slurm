#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=workq
#SBATCH --time=00:30:00 
#SBATCH -A k1033
#SBATCH --job-name=demo
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j-slurm.out
#SBATCH --error=%x-%j-slurm.err 

##########################################################
# remember you have to execute this script from your     #
# /scratch/$USER  folder                                 #
##########################################################


export LC_ALL=C.UTF-8
export LANG=C.UTF-8

##########################################################
#  Load the modules you need for your job               #
#  and swap the PrgEnv module if needed                 #
##########################################################
module swap PrgEnv-cray PrgEnv-intel
#module load intelpython3/2022.0.2.155 pytorch/1.8.0

############################################################
## Load the conda base and activate the conda environment ##
############################################################
############################################################
## export the path to the conda base and conda_cache       #
############################################################
export ENV_PREFIX=/project/k1033/${USER}
export CONDA_PKGS_DIRS=$ENV_PREFIX/conda_cache
############################################################ 
## activate conda base from the command line               #
############################################################
source $ENV_PREFIX/miniconda3/bin/activate $ENV_PREFIX/install_miniconda_shaheen/env

####################################################################
# Or be very explicit and use the full path to the activate script #
# like in the line below                                           #
####################################################################
#source /project/k1033/barradd/install_miniconda_shaheen/miniconda3/bin/activate /project/k1033/barradd/install_miniconda_shaheen/env

# setup the environment
export SCRATCH_DIR=/scratch/${USER}
export CODE_SERVER_CONFIG_FOLDER=${SCRATCH_DIR}/.config/code-server
export CODE_SERVER_CONFIG=${CODE_SERVER_CONFIG_FOLDER}/config.yaml
export XDG_CONFIG_HOME=${SCRATCH_DIR}/tmpdir
export CODE_SERVER_EXTENSIONS=${SCRATCH_DIR}/code-server/extensions
PROJECT_DIR="$PWD"
PATH="${SCRATCH_DIR}/.local/bin:$PATH"
mkdir -p ${CODE_SERVER_CONFIG_FOLDER} ${SCRATCH_DIR}/data ${XDG_CONFIG_HOME} ${CODE_SERVER_EXTENSIONS} 

# setup ssh tunneling 
COMPUTE_NODE=$(hostname -s) 
CODE_SERVER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
submit_host=${SLURM_SUBMIT_HOST} 
gateway=${EPROXY_LOGIN}

# Check is config.yaml is there , other wise create it 

if [ ! -f $CODE_SERVER_CONFIG ] ; then 
    touch ${CODE_SERVER_CONFIG}
    echo "bind-addr: 127.0.0.1:8080" >> ${CODE_SERVER_CONFIG} 
    echo "auth: password" >> ${CODE_SERVER_CONFIG}
    echo "password: 10DowningStreet" >> ${CODE_SERVER_CONFIG}
    echo "cert: false" >> ${CODE_SERVER_CONFIG}
fi ; 


echo "
To connect to the compute node ${COMPUTE_NODE} on Shaheen running your Code Server.
Copy the following two lines in a new terminal one after another to create a secure SSH tunnel between your computer and Shaheen compute node.
ssh -L ${CODE_SERVER_PORT}:localhost:${CODE_SERVER_PORT} ${USER}@${submit_host}.hpc.kaust.edu.sa 
ssh -L ${CODE_SERVER_PORT}:${COMPUTE_NODE}:${CODE_SERVER_PORT} ${USER}@${gateway}

Next, you need to copy the url provided below and paste it into the browser 
on your local machine.

localhost:${CODE_SERVER_PORT}

" >&2

# launch code server
code-server --auth none --user-data-dir=${SCRATCH_DIR}/data --bind-addr ${COMPUTE_NODE}:${CODE_SERVER_PORT} --extensions-dir=${CODE_SERVER_EXTENSIONS} "$PROJECT_DIR"
