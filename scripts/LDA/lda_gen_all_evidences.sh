#!/bin/bash

# Check if the arguments are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <lda_script> <input_dir> <output_dir> <shrinkage> <scaler>"
    exit 1
fi

# Set the arguments
lda_script="$1"
input_dir="$2"
out_dir="$3"
shrinkage="$4"
scaler="$5"

# List of subjects
dsbl_sbjs=$(seq 1 4)
ctrl_sbjs=$(seq 6 9)
sbjs=($dsbl_sbjs $ctrl_sbjs)

for sbj in "${sbjs[@]}"; do
    python $lda_script -s $sbj -id $input_dir -o $out_dir/LDA/S$sbj/values.pkl -sh $shrinkage -sc $scaler
    echo "Subject $sbj done"
done
