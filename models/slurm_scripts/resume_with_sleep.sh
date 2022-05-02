#!/bin/bash

#sleep 5400
#export CHECKPOINT_DIR=/brtx/606-nvme1/estengel/annotator_uncertainty/models/vilt_finetuned_bce_ce/
#export TRAINING_CONFIG=config/base/vilt/vilt_bce_ce_distributed.jsonnet
export CHECKPOINT_DIR=/brtx/603-nvme1/estengel/annotator_uncertainty/models/vilt_finetuned_bce/
export TRAINING_CONFIG=config/base/vilt/vilt_bce_distributed.jsonnet

sbatch slurm_scripts/resume.sh --export 
