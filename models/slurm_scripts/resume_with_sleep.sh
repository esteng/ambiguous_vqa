#!/bin/bash

sleep 900
export CHECKPOINT_DIR=/brtx/605-nvme1/estengel/annotator_uncertainty/models/vilt_bce_ce_3_layer_double/
#export CHECKPOINT_DIR=/brtx/605-nvme1/estengel/annotator_uncertainty/models/vilt_bce_ce_3_layer/
export TRAINING_CONFIG=${CHECKPOINT_DIR}/ckpt/config.json 
sbatch slurm_scripts/resume.sh --export 
