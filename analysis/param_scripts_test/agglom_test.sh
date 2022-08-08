#!/bin/bash 

python text_cluster_search.py \
    --agglom \
    --method ward \
    --t 7.5  \
    --embedder path \
    --checkpoint-dir /brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit/ \
    --pooler max \
    --test 
