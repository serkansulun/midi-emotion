#!/bin/bash
#

##SBATCH --partition=cpu_14cores
##SBATCH --qos=cpu_14cores

#SBATCH --partition=gpu_min80GB
#SBATCH --qos=gpu_min80GB

#SBATCH --output=log_slurm/%j.txt          # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --job-name=regression             # Job name​

cd src
~/.conda/envs/torch/bin/python train.py --regression
