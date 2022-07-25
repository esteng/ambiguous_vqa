#!/bin/bash 

# need CHECKPOINT_DIR,  TEST_DATA

echo "visbile: $CUDA_VISIBLE_DEVICES" 

#for shard in  0 1 2 3 4 5
for shard in missing
do 
    export TEST_DATA=/brtx/603-nvme2/estengel/annotator_uncertainty/vqa/dev_test_data_line_by_line/test/${shard}/
    sbatch slurm_scripts/predict.sh 
    sbatch slurm_scripts/predict_forced.sh 
    sleep 300
done 

