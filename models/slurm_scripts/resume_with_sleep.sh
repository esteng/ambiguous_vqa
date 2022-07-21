#!/bin/bash

sleep 21600;
export CHECKPOINT_DIR=/brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit_train_vilt_mlm/
export TRAINING_CONFIG=${CHECKPOINT_DIR}/ckpt/config.json 
sbatch slurm_scripts/resume.sh --export 
