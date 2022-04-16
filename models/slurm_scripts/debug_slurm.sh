#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/debug.out
#SBATCH -p brtx6

# need CHECKPOINT_DIR, TRAINING_CONFIG

echo $CHECKPOINT_DIR
echo $TRAINING_CONFIG
