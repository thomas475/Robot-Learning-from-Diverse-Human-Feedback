#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=diffusion_ql
#SBATCH --gres=gpu:4

module load devel/miniconda
module load devel/cuda devel/cudnn
conda activate rlhf_cuda
cd /home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 diffusion_ql.py --config "/home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/baseline_gpu.yaml"