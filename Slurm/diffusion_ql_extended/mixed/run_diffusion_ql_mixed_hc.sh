#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=40
#SBATCH --job-name=diffusion_ql

module load devel/miniconda
conda activate rlhf
cd /home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 diffusion_ql.py --config "/home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/mixed_hc.yaml" --log_dir $TMPDIR