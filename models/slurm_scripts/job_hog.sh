#!/bin/bash 

#SBATCH -o /home/estengel/annotator_uncertainty/logs/hog.out
#SBATCH -p brtx6
#SBATCH --gpus=1

echo "visible: $CUDA_VISIBLE_DEVICES"
sleep 300
