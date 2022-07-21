#!/bin/bash 

# need CHECKPOINT_DIR,  TEST_DATA

echo "visbile: $CUDA_VISIBLE_DEVICES" 

for shard in  5 6 7 8 
do 
    export TEST_DATA=/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/filtered_line_by_line/${shard}/
    #sbatch slurm_scripts/predict.sh 
    sbatch slurm_scripts/predict_forced.sh 
    sleep 300
done 

