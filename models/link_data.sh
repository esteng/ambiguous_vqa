#!/bin/bash

for file in /brtx/603-nvme2/estengel/annotator_uncertainty/vqa/val2014/*.jpg
do
    ln -s ${file} /brtx/603-nvme2/estengel/annotator_uncertainty/vqa/balanced_real/
done
