#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=40
#SBATCH --job-name=diffusion_ql

module load devel/miniconda
conda activate rlhf
cd [base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 diffusion_ql.py --config "[base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/mixed_baseline.yaml" --log_dir $TMPDIR