#!/bin/bash

#$ -j yes
#$ -N search_job
#$ -o /home/jgualla1/decomp/decomp/jimena_work/test_5.txt
#$ -l 'mem_free=1M, h_rt=01:00:00'
#$ -m ae -M jgualla1@jh.edu
#4 -cwd

python3 /home/jgualla1/decomp/decomp/jimena_work/query_test.py
