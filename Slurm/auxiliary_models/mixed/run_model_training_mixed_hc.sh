#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=40
#SBATCH --job-name=auxiliary_model

module load devel/miniconda
conda activate rlhf
cd [base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/rlhf
python3 train_model.py --config "[base-directory]/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/configs/experiments/auxiliary_models/mixed_hc.yaml" --log_dir $TMPDIR