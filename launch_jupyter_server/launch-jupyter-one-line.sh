#!/bin/bash
# Activate the environment and execute the commands within a subshell
(
    eval "$(conda shell.bash hook)"
    # Load and run packages
    module load machine_learning
    # or activate the conda environment
    #export ENV_PREFIX=$PWD/env
    #conda activate $ENV_PREFIX
    # module load cudnn/8.8.1-cuda11.8.0
    jupyter lab --no-browser --ip="$(hostname)".ibex.kaust.edu.sa

)
