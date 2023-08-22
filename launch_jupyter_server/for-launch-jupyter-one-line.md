# Instructions

To run a specific command on a computing cluster using Slurm job management, follow these steps:

1. Open a terminal window.

2. Use the following command to submit a job using the `srun` command and specify the desired resource allocation options:
    
        ```bash
        srun --gpus=1 --mem=100G --cpus-per-task=24 -C v100 --time=00:30:00 --resv-ports=1 --pty /bin/bash -l launch-jupyter-one-line.sh
        ```
    

    Here's a breakdown of the options used:

- `--gpus=1`: Request 1 GPU for the job.
- `--mem=100G`: Request 100GB of memory.
- `--cpus-per-task=24`: Request 24 CPU cores per task.
- `-C v100`: Request a compute node with NVIDIA V100 GPUs.
- `--time=00:30:00`: Request a maximum job runtime of 30 minutes.
- `--resv-ports=1`: Reserve a port for the job.
- `--pty`: Allocate a pseudo terminal for the job.
- `/bin/bash -l launch-jupyter-one-line.sh`: Run the `launch-jupyter-one-line.sh` script in a Bash shell with the login environment.

3. After executing the command, the job will be submitted to the cluster and will run according to the specified resource allocation and script instructions. The job will be assigned a job ID, which will be displayed in the terminal window. You can use this job ID to monitor the job's progress and check its status.

For more information on Slurm commands and options, refer to the official documentation.
