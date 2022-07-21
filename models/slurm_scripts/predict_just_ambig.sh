#!/bin/bash 

# need CHECKPOINT_DIR,  TEST_DATA

echo "visbile: $CUDA_VISIBLE_DEVICES" 

export TEST_DATA=/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/just_ambig_line_by_line
sbatch slurm_scripts/predict.sh 
sbatch slurm_scripts/predict_forced.sh 

