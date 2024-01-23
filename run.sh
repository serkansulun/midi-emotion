#!/bin/bash
#
#SBATCH --partition=cpu_14cores           # Partition (check with "$sinfo")
#SBATCH --output=log_slurm/%j.txt          # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --qos=cpu_14cores                 # (Optional) QoS level (check with "$sacctmgr show qos")
#SBATCH --job-name=processing             # Job name​

cd src/data
~/.conda/envs/torch/bin/python preprocess_pianorolls.py

# cp -r ~/data/midi/lpd_5/lakh_pianoroll_dataset_5_full /nas-ctm01/datasets/public/

# ~/.conda/envs/torch/bin/python train.py --debug --log_step 100 --n_layer 8 --max_step 300

# for condition in continuous_token continuous_concat discrete_token
# do
# ~/.conda/envs/torch/bin/python generate.py --model_dir $condition --conditioning $condition \
#     --valence -0.8 -0.8 0.8 0.8 --arousal -0.8 0.8 -0.8 0.8 --quiet --num_runs 16
# echo "${condition} done"
# done