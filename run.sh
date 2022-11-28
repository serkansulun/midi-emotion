#!/bin/bash
#
#SBATCH --partition=rtx6000_24GB           # Partition (check with "$sinfo")
#SBATCH --output=log_slurm/%j.txt          # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --qos=normal                 # (Optional) QoS level (check with "$sacctmgr show qos")
#SBATCH --job-name=midi             # Job name​

cd src
# ~/.conda/envs/torch/bin/python train.py --debug --log_step 100 --n_layer 8 --max_step 300

for condition in continuous_token continuous_concat discrete_token
do
~/.conda/envs/torch/bin/python generate.py --model_dir $condition --conditioning $condition \
    --valence -0.8 -0.8 0.8 0.8 --arousal -0.8 0.8 -0.8 0.8 --quiet --num_runs 16
echo "${condition} done"
done