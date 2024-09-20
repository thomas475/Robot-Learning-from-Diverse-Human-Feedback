#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=40
#SBATCH --job-name=iql

module load devel/miniconda
conda activate rlhf
cd [base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/algorithms/offline/
python3 iql_p.py --config "[base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/basic/mixed_se.yaml" --log_dir $TMPDIR