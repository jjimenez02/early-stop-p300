#!/bin/bash

# Directories
fst_coin_dir="First Coincidence"
fix_param_dir="Fixed Parameters"
gl_bal_dir="Gain - Loss balance"
baseline_dir="Baseline"
out_dir="Results/Subjects Comparison"

# List of tests
tests=("mannwhitneyu" "ttest_ind" "ranksums" "ks_2samp" "ttest_welch")

# Removing old results
trash-put "$out_dir"/*

# Tests plots
all_tests(){
    # Without Bonferroni
    # Fixed parameters scripts
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Stat_Test/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Stat_Test/LDA/$1 &
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Acc_Evid_Stat_Test/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Acc_Evid_Stat_Test/LDA/$1 &
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1 &
    # Gain - Loss balance scripts
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Stat_Test/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Stat_Test/LDA/$1 &
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Evid_Stat_Test/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Acc_Evid_Stat_Test/LDA/$1 &
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Acc_Avg_Diff_Stat_Test/LDA/$1 &
    
    # With Bonferroni
    # Fixed parameters scripts
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Stat_Test_Bonferroni/LDA/$1 &
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1 &
    python "$fix_param_dir"/sbj_comparison_fixed_params.py -id "$fix_param_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$fix_param_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1 &
    # Gain - Loss balance scripts
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Stat_Test_Bonferroni/LDA/$1 &
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Acc_Evid_Stat_Test_Bonferroni/LDA/$1 &
    python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1 -od "$out_dir"/"$gl_bal_dir"/Acc_Avg_Diff_Stat_Test_Bonferroni/LDA/$1 &
}

# Plot all bar-plots
# Baseline
python "$fst_coin_dir"/sbj_comparison_fst_coincidence.py -id "$baseline_dir"/Baseline/LDA -od "$out_dir"/"$baseline_dir"/LDA &

# First Coincidence
python "$fst_coin_dir"/sbj_comparison_fst_coincidence.py -id "$fst_coin_dir"/Fixed_Stop/LDA -od "$out_dir"/"$fst_coin_dir"/Fixed_Stop/LDA &
python "$fst_coin_dir"/sbj_comparison_fst_coincidence.py -id "$fst_coin_dir"/Acc_Evid/LDA -od "$out_dir"/"$fst_coin_dir"/Acc_Evid/LDA &
python "$fst_coin_dir"/sbj_comparison_fst_coincidence.py -id "$fst_coin_dir"/Acc_Avg_Diff/LDA -od "$out_dir"/"$fst_coin_dir"/Acc_Avg_Diff/LDA &

# Gain-Loss Balance
python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Evid/LDA -od "$out_dir"/"$gl_bal_dir"/Acc_Evid/LDA &
python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Fixed_Stop/LDA -od "$out_dir"/"$gl_bal_dir"/Fixed_Stop/LDA &
python "$gl_bal_dir"/sbj_comparison_gain_loss_balance.py -id "$gl_bal_dir"/Acc_Avg_Diff/LDA -od "$out_dir"/"$gl_bal_dir"/Acc_Avg_Diff/LDA &

# Statistical tests
for test in "${tests[@]}"; do
    all_tests $test
done

