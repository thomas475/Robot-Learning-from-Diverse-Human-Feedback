# kitchen-complete-v0
sbatch -p single complete/run_model_training_complete_hc.sh
sbatch -p single complete/run_model_training_complete_he.sh
sbatch -p single complete/run_model_training_complete_hk.sh
sbatch -p single complete/run_model_training_complete_sc.sh
sbatch -p single complete/run_model_training_complete_se.sh

# kitchen-mixed-v0
sbatch -p single mixed/run_model_training_mixed_hc.sh
sbatch -p single mixed/run_model_training_mixed_he.sh
sbatch -p single mixed/run_model_training_mixed_hk.sh
sbatch -p single mixed/run_model_training_mixed_sc.sh
sbatch -p single mixed/run_model_training_mixed_se.sh

# kitchen-partial-v0
sbatch -p single partial/run_model_training_partial_hc.sh
sbatch -p single partial/run_model_training_partial_he.sh
sbatch -p single partial/run_model_training_partial_hk.sh
sbatch -p single partial/run_model_training_partial_sc.sh
sbatch -p single partial/run_model_training_partial_se.sh