#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/train3.out 
#SBATCH -p brtx6
#SBATCH --gpus=4

# need CHECKPOINT_DIR, TRAINING_CONFIG

echo "visbile: $CUDA_VISIBLE_DEVICES" 
./scripts/main.sh -a train 

