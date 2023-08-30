# This script assumes that miniconda is installed on WekaIO 
# thus you can modify the PREFIX 

# USAGE: source source_conda.sh

# establish the PREFIX
export PREFIX=/ibex/scratch/$USER

hostname_sys=$(hostname)
case $hostname_sys in
# GPU login node
  login510-27 | login510-29 | login510-22   )
    export CONDA_PKGS_DIRS=${PREFIX}/conda_cache
    source ${PREFIX}/miniconda3/bin/activate
    ;;
# CPU login node
  login509-02-l | login509-02-r )
    export CONDA_PKGS_DIRS=${PREFIX}/conda_cache
    source ${PREFIX}/miniconda3/bin/activate
    ;;
# for interactive sessions
  *)
    echo "export CONDA_PKGS_DIRS=/ibex/user/${PREFIX}/conda_cache"
    echo "source /ibex/user/${PREFIX}/miniconda3/bin/activate"
    ;;
esac
