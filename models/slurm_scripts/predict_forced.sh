#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/predf.out
#SBATCH -p brtx6
#SBATCH --gpus=1

# need CHECKPOINT_DIR, TRAINING_CONFIG, TEST_DATA

echo "visbile: $CUDA_VISIBLE_DEVICES" 
./scripts/main.sh -a predict_forced

