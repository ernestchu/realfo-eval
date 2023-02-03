#!/bin/bash

declare -a arr=("DF" "F2F" "FS" "NT")
COMP=c23

set -x

for i in "${arr[@]}"
do
    accelerate launch eval.py $i
done

