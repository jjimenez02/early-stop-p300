#!/bin/bash

# Check if the arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_data_path>"
    exit 1
fi

# Constants
N_BEST=3
METRICS=("GAIN" "LOSS" "CONS" "GAIN+CONS÷2")
# Groups
HYPER_PARAM_GRP="Hyper-parameterisable"
GL_BAL_GRP="Gain - Loss balance"
# Subgroups
FST_COIN_SUBGRP="First Coincidence"
STAT_TEST_BONF_SUBGRP="ST - Bonferroni"
STAT_TEST_NON_BONF_SUBGRP="ST - Non Bonferroni"

# Arguments
results_path="$1"

# Directories
plots_dir="scripts/Plots"
output_dir="Results/Methods Comparison"

# First execute the subject comparisons' plots
./scripts/Plots/execute_all_sbjs_comparisons.sh

# Remove old results
trash-put "$output_dir"/*

# Now obtain all the results useful for the thesis
# Generic function to obtain them all
all_plots() {
    python "$plots_dir"/auc_bar_plots_script.py -i "$1" -od "$output_dir"/AUC_BarPlots/"$2" -m "$3" -n $N_BEST -reg "$4" -bm
    python "$plots_dir"/sbjs_bar_plots_script.py -i "$1" -od "$output_dir"/BarPlots/"$2" -m "$3" -n $N_BEST -reg "$4" -bm
    python "$plots_dir"/sbjs_plots_script.py -i "$1" -od "$output_dir"/CurvesPlots/"$2" -m "$3" -n $N_BEST -reg "$4" -bm
    python "$plots_dir"/sbjs_spider_plot_script.py -i "$1" -od "$output_dir"/SpiderPlots/"$2" -m "$3" -n $N_BEST -reg "$4" -bm
}

for metric in "${METRICS[@]}"; do
    # Hyper-parameterisable groups
    # First coincidence methods
    all_plots "$results_path" "$HYPER_PARAM_GRP/$FST_COIN_SUBGRP/$metric" "$metric" ".*First Coincidence.*" &
    # Statistical tests
    # With Bonferroni
    all_plots "$results_path" "$HYPER_PARAM_GRP/$STAT_TEST_BONF_SUBGRP/$metric" "$metric" ".*Stat Test - Fixed alpha.*Bonferroni: True.*" &
    # Without Bonferroni
    all_plots "$results_path" "$HYPER_PARAM_GRP/$STAT_TEST_NON_BONF_SUBGRP/$metric" "$metric" ".*Stat Test - Fixed alpha.*Bonferroni: False.*" &

    # Gain-Loss balance optimisation groups
    # First coincidence methods
    all_plots "$results_path" "$GL_BAL_GRP/$FST_COIN_SUBGRP/$metric" "$metric" ".*G-L Bal$" &
    # Statistical tests
    # With Bonferroni
    all_plots "$results_path" "$GL_BAL_GRP/$STAT_TEST_BONF_SUBGRP/$metric" "$metric" ".*Stat Test - G-L Bal.*Bonferroni: True.*" &
    # Without Bonferroni
    all_plots "$results_path" "$GL_BAL_GRP/$STAT_TEST_NON_BONF_SUBGRP/$metric" "$metric" ".*Stat Test - G-L Bal.*Bonferroni: False.*" &
done
