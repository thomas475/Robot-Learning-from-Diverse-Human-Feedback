#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=td3_bc_p

module load devel/miniconda
conda activate rlhf
cd /home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 td3_bc_p.py --config "/home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/baseline.yaml"