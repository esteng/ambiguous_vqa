#!/bin/bash

for pooler in mean max 
do 
    python text_cluster_search.py \
        --kmeans \
        --embedder path \
        --checkpoint-dir /brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit/ \
        --pooler ${pooler} \
        --test 
done
