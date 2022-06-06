#!/bin/bash

sleep 648
export CHECKPOINT_DIR=/brtx/602-nvme1/estengel/annotator_uncertainty/models/vilt_mlm/
export TRAINING_CONFIG=${CHECKPOINT_DIR}/ckpt/config.json 
sbatch slurm_scripts/resume.sh --export 
