#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=4000
#SBATCH --job-name=train_reward_model

module load devel/miniconda
conda activate rlhf
cd /home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/rlhf
python3 train_reward_model.py