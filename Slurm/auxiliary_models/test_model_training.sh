#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=40
#SBATCH --job-name=auxiliary_model

module load devel/miniconda
conda activate rlhf
cd /home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Clean-Offline-RLHF/rlhf
python3 train_model.py --config "/home/kit/anthropomatik/px6987/Robot-Learning-from-Diverse-Human-Feedback/Slurm/auxiliary_models/test_config.yaml" --log_dir $TMPDIR