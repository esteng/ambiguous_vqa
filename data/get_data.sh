#!/bin/bash

DATA_DIR=$1

# VQA
mkdir -p ${DATA_DIR}/vqa/
echo "Getting VQA train" 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P ${DATA_DIR}/vqa/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P ${DATA_DIR}/vqa/
echo "Getting VQA dev" 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P ${DATA_DIR}/vqa/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P ${DATA_DIR}/vqa/

