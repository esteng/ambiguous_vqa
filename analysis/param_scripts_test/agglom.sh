#!/bin/bash

for pooler in mean max 
do 
    for method in ward complete average single
    do
        #for num_clust in 2 3
        for dist in 7.0 7.5 8.0 8.5 
        do 
            python text_cluster_search.py \
                --agglom \
                --method ${method} \
                --t ${dist} \
                --embedder path \
                --checkpoint-dir /brtx/602-nvme1/estengel/annotator_uncertainty/models/img2q_t5_base_no_limit/ \
                --pooler ${pooler} 
        done 
    done
done
