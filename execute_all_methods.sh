#!/bin/bash

# Check if the arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir>"
    exit 1
fi

# Set the arguments
input_dir="$1"

# List of subjects
dsbl_sbjs=$(seq 1 4)
ctrl_sbjs=$(seq 6 9)
sbjs=($dsbl_sbjs $ctrl_sbjs)

# Directories
fst_coin_dir="First Coincidence"
fix_param_dir="Fixed Parameters"
gl_bal_dir="Gain - Loss balance"
baseline_dir="Baseline"

# Removing old results
# First coincidence
trash-put "$fst_coin_dir"/Fixed_Stop
trash-put "$fst_coin_dir"/Acc_Evid
trash-put "$fst_coin_dir"/Acc_Avg_Diff

# Fixed parameters
trash-put "$fix_param_dir"/Stat_Test
trash-put "$fix_param_dir"/Acc_Evid_Stat_Test
trash-put "$fix_param_dir"/Stat_Test_Bonferroni
trash-put "$fix_param_dir"/Acc_Evid_Stat_Test_Bonferroni
trash-put "$fix_param_dir"/Acc_Avg_Diff_Stat_Test
trash-put "$fix_param_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni

# Gain-Loss balance
trash-put "$gl_bal_dir"/Fixed_Stop
trash-put "$gl_bal_dir"/Acc_Evid
trash-put "$gl_bal_dir"/Acc_Avg_Diff
trash-put "$gl_bal_dir"/Stat_Test
trash-put "$gl_bal_dir"/Acc_Evid_Stat_Test
trash-put "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test
trash-put "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni
trash-put "$gl_bal_dir"/Stat_Test_Bonferroni
trash-put "$gl_bal_dir"/Acc_Evid_Stat_Test_Bonferroni

# Baseline
trash-put "$baseline_dir"/Baseline

# List of tests
tests=("mannwhitneyu" "ttest_ind" "ranksums" "ks_2samp" "ttest_welch")

# Tests execution
all_tests(){
    # Without Bonferroni
    # Fixed parameters scripts
    python "$fix_param_dir"/stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Stat_Test/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1
    python "$fix_param_dir"/acc_evid_stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Acc_Evid_Stat_Test/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1 &
    python "$fix_param_dir"/acc_avg_diff_stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1 &
    # Gain - Loss balance scripts
    python "$gl_bal_dir"/stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Stat_Test/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 &
    python "$gl_bal_dir"/acc_evid_stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Evid_Stat_Test/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 &
    python "$gl_bal_dir"/acc_avg_diff_stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 &

    # With Bonferroni
    # Fixed parameters scripts
    python "$fix_param_dir"/stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Stat_Test_Bonferroni/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1 -b
    python "$fix_param_dir"/acc_evid_stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1 -b &
    python "$fix_param_dir"/acc_avg_diff_stat_test_gain_loss_metrics_script.py -s $sbj -id $input_dir -o "$fix_param_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1/S$sbj/results.pkl -sh -1 -sc Original -tmax 20 -a 0.05 -st $1 -b &
    # Gain - Loss balance scripts
    python "$gl_bal_dir"/stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Stat_Test_Bonferroni/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 -b &
    python "$gl_bal_dir"/acc_evid_stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 -b &
    python "$gl_bal_dir"/acc_avg_diff_stat_test_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1/S$sbj -sh -1 -sc Original -tmax 20 -st $1 -b &
}

for sbj in "${sbjs[@]}"; do
    # Baseline script
    python "$baseline_dir"/baseline_script.py -s $sbj -id $input_dir -o "$baseline_dir"/Baseline/LDA/S$sbj/results.pkl -sh -1 -sc Original -tmax 20

    # First coincidence scripts
    python "$fst_coin_dir"/fixed_stop_optimization_script.py -s $sbj -id $input_dir -o "$fst_coin_dir"/Fixed_Stop/LDA/S$sbj/results.pkl -sh -1 -sc Original -tmax 20
    python "$fst_coin_dir"/acc_evid_threshold_optimization_script.py -s $sbj -id $input_dir -o "$fst_coin_dir"/Acc_Evid/LDA/S$sbj/results.pkl -sh -1 -sc Original -tmax 20
    python "$fst_coin_dir"/acc_avg_diff_threshold_optimization_script.py -s $sbj -id $input_dir -o "$fst_coin_dir"/Acc_Avg_Diff/LDA/S$sbj/results.pkl -sh -1 -sc Original -tmax 20

    # Gain - Loss balance scripts
    python "$gl_bal_dir"/fixed_stop_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Fixed_Stop/LDA/S$sbj -sh -1 -sc Original -tmax 20 &
    python "$gl_bal_dir"/acc_evid_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Evid/LDA/S$sbj -sh -1 -sc Original -tmax 20 &
    python "$gl_bal_dir"/acc_avg_diff_gain_loss_balance_optimization_script.py -s $sbj -id $input_dir -o "$gl_bal_dir"/Acc_Avg_Diff/LDA/S$sbj -sh -1 -sc Original -tmax 20 &

    # Statistical tests
    for test in "${tests[@]}"; do
        all_tests $test
    done

    echo "Subject $sbj sent"
done
