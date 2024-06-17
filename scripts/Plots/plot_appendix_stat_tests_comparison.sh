#!/bin/bash

# Check if the arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <results_data_path> <out_dir>"
    exit 1
fi

# Constants
METRICS=("GAIN" "LOSS" "CONS" "GAIN+CONS÷2")
REGEXP_A=".*Stat Test.*Fixed alpha:0.05.*Bonferroni: True"
REGEXP_B=".*Stat Test.*Fixed alpha:0.05.*Bonferroni: False"

# Arguments
results_path="$1"
output_dir="$2"

# Directories
plots_dir="scripts/Plots"

# Remove old results
trash-put "$output_dir"/*

# Now obtain all the results useful for the thesis
# Generic function to obtain them all
all_plots() {
    python "$plots_dir"/auc_bar_plots_script.py -i "$1" -od "$5"/AUC_BarPlots/"$2" -m "$3" -reg "$4" -bm
}

for metric in "${METRICS[@]}"; do
    all_plots "$results_path" "$metric" "$metric" "$REGEXP_A" "$output_dir/BF_True" &
done

for metric in "${METRICS[@]}"; do
    all_plots "$results_path" "$metric" "$metric" "$REGEXP_B" "$output_dir/BF_False" &
done
