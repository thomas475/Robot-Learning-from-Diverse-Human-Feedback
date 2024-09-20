#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=iql
#SBATCH --gres=gpu:4

module load devel/miniconda
module load devel/cuda devel/cudnn
conda activate rlhf_cuda
cd [base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 iql_p.py --config "[base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/baseline_gpu.yaml" --log_dir $TMPDIR