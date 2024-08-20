#!/bin/bash
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50G
#SBATCH --output=rstudio-server.job%j.out
#SBATCH --error=rstudio-server.job%j.error

# load singularity module

module load singularity/3.9.7
module load rl9-cpustack

# Pull singularity image
singularity pull library://mgharawy97/collection/rstudio-server.sif
###################################


# Create temporary directory to be populated with directories to bind-mount in the container
# where writable file systems are necessary. Adjust path as appropriate for your computing environment.
workdir=$(python -c 'import tempfile; print(tempfile.mkdtemp())')

mkdir -p -m 700 ${workdir}/run ${workdir}/tmp ${workdir}/var/lib/rstudio-server
cat > ${workdir}/database.conf <<END
provider=sqlite
directory=/var/lib/rstudio-server
END

# Set OMP_NUM_THREADS to prevent OpenBLAS (and any other OpenMP-enhanced
# libraries used by R) from spawning more threads than the number of processors
# allocated to the job.
#
# Set R_LIBS_USER to a path specific to rocker/rstudio to avoid conflicts with
# personal libraries from any R installation in the host environment

cat > ${workdir}/rsession.sh <<END
#!/bin/bash
export OMP_NUM_THREADS=${SLURM_JOB_CPUS_PER_NODE}
export R_LIBS_USER=${HOME}/R/rocker-rstudio/4.0.5
#export R_LIBS=/home/$USER/local/R4.3.0_libs.gnu
exec rsession "\${@}"
END

chmod +x ${workdir}/rsession.sh

#export SINGULARITY_BIND="${workdir}/run:/run,${workdir}/tmp:/tmp,${workdir}/database.conf:/etc/rstudio/database.conf,${workdir}/rsession.sh:/etc/rstudio/rsession.sh,${workdir}/var/lib/rstudio-server:/var/lib/rstudio-server,/sw:/sw,/ibex/sw:/ibex/sw:/ibex/user:/ibex/user"
export SINGULARITY_BIND="${workdir}/run:/run,${workdir}/tmp:/tmp,${workdir}/database.conf:/etc/rstudio/database.conf,${workdir}/rsession.sh:/etc/rstudio/rsession.sh,${workdir}/var/lib/rstudio-server:/var/lib/rstudio-server,/ibex/sw:/ibex/sw,/ibex/user:/ibex/user,/sw:/sw"

# Do not suspend idle sessions.
# Alternative to setting session-timeout-minutes=0 in /etc/rstudio/rsession.conf
# https://github.com/rstudio/rstudio/blob/v1.4.1106/src/cpp/server/ServerSessionManager.cpp#L126
export SINGULARITYENV_RSTUDIO_SESSION_TIMEOUT=0

export SINGULARITYENV_USER=$(id -un)
# get unused socket per https://unix.stackexchange.com/a/132524
# tiny race condition between the python & singularity commands
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

ssh  -L ${PORT}:${HOSTNAME}:${PORT} ${SINGULARITYENV_USER}@ilogin.ibex.kaust.edu.sa

and point your web browser to http://localhost:${PORT}

When done using RStudio Server, terminate the job by:

1. Exit the RStudio Session ("power" button in the top right corner of the RStudio window)
2. Issue the following command on the login node:

    scancel -f ${SLURM_JOB_ID}

END

export SINGULARITYENV_PATH=$PATH:/usr/lib/rstudio-server/bin

# Modify the value of next line to load the modules to use with Rstudio.
echo "here"
module avail bioconductor
export modules="bioconductor/3.19/R-4.4.0"

singularity exec rstudio-server.sif_latest.sif \
    bash -c "module load ${modules}  && rm -rf ~/.local/share/rstudio/ && rserver --www-port=${PORT} \
            --auth-none=1 \
            --auth-pam-helper-path=pam-helper \
            --auth-stay-signed-in-days=30 \
            --auth-timeout-minutes=0 \
            --server-user=$(whoami) \
            --server-daemonize=0 \
            --auth-minimum-user-id=0 \
            --rsession-path=/etc/rstudio/rsession.sh"

printf 'rserver exited' 1>&2