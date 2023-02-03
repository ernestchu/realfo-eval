#!/bin/bash

declare -a arr=("DF" "F2F" "FS" "NT")
COMP=c23
DATA="/home/ernestchu/scratch4/OriginalDatasets/FF/"

set -x

for i in "${arr[@]}"
do
    accelerate launch eval.py $i --data_root $DATA
done

