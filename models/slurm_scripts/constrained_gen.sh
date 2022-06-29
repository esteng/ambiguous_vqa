#!/bin/bash 

config=$CHECKPOINT_DIR/ckpt/config.json
cp ${config} ${CHECKPOINT_DIR}/ckpt/config_orig.json 
python scripts/edit_config_constraint.py ${config} 

cd $CHECKPOINT_DIR/ckpt
cp best.th weights.th 
tar -czvf model.tar.gz weights.th config.json vocabulary 
cd ~/annotator_uncertainty/models

sbatch slurm_scripts/vqa_eval.sh --export 
