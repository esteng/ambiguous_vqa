#!/bin/bash

export CHECKPOINT_DIR=/brtx/603-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_text_baseline
export TRAINING_CONFIG=${CHECKPOINT_DIR}/ckpt/config.json 
sbatch slurm_scripts/resume.sh --export 
