#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time=02:00:00
#SBATCH --mem=100M
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]

#run the application:
Myapp

