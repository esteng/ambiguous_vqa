#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/train.out 
#SBATCH -p brtx6
#SBATCH --gpus=4

# need CHECKPOINT_DIR, TRAINING_CONFIG

./scripts/main.sh -a train 

