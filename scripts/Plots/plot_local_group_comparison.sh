#!/bin/bash

# Check if the arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <results_data_path> <out_dir>"
    exit 1
fi

# Constants
METRICS=("GAIN" "LOSS" "CONS" "GAIN+CONS÷2")
REGEXP="Acc Avg Diff Stat Test - Fixed alpha:0.05 - ttest_ind - Bonferroni: False|Acc Avg Diff Stat Test - Fixed alpha:0.05 - ttest_welch - Bonferroni: False|Acc Avg Diff Stat Test - Fixed alpha:0.05 - mannwhitneyu - Bonferroni: False|Acc Avg Diff Stat Test - Fixed alpha:0.05 - ttest_ind - Bonferroni: True|Acc Avg Diff Stat Test - Fixed alpha:0.05 - ttest_welch - Bonferroni: True|Acc Evid Stat Test - Fixed alpha:0.05 - ttest_ind - Bonferroni: True|Acc Evid Threshold - First Coincidence|Acc Avg Diff Threshold - First Coincidence|Fixed Stop - First Coincidence"

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
    python "$plots_dir"/auc_bar_plots_script.py -i "$1" -od "$output_dir"/AUC_BarPlots/"$2" -m "$3" -reg "$4" -bm
    python "$plots_dir"/sbjs_bar_plots_script.py -i "$1" -od "$output_dir"/BarPlots/"$2" -m "$3" -reg "$4" -bm
    python "$plots_dir"/sbjs_plots_script.py -i "$1" -od "$output_dir"/CurvesPlots/"$2" -m "$3" -reg "$4" -bm
    python "$plots_dir"/sbjs_spider_plot_script.py -i "$1" -od "$output_dir"/SpiderPlots/"$2" -m "$3" -reg "$4" -bm
}

for metric in "${METRICS[@]}"; do
    all_plots "$results_path" "$metric" "$metric" "$REGEXP" &
done
