#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/pred.out
#SBATCH -p brtx6
#SBATCH --gpus=4

# need CHECKPOINT_DIR, TEST_DATA

echo "visbile: $CUDA_VISIBLE_DEVICES" 
./scripts/main.sh -a predict

