#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/log.out
#SBATCH -p brtx6
#SBATCH --gpus=1

echo "submitted"
./scripts/main.sh -a train
