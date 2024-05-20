'''
This script will obtain the gain & loss metrics
by performing a specific statistical test over
the averaged differences between predicted
stimulus and with a particular significance level
specified via parameters. The results will be
averaged along every session out for a single
subject.

This method does not consider any.
inter-subject feature.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 25/04/2024
'''

import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from metrics import (gain,
                     loss)
from constants import (SEED,
                       HOFF_SUBJECTS,
                       HOFF_SESSIONS,
                       HOFF_RUNS,
                       AVAILABLE_TESTS)
from early_stop import (stat_test,
                        stat_test_chosen_stim_es_trials)
from data_utils import (load_data,
                        sklearn_reshape)
from transformers import choose_scaler
from voting_utils import (soft_voting,
                          soft_voting_reshape,
                          avg_diffs,
                          sort_stim)
from report_utils import (es_extra_vals,
                          es_save_vals,
                          es_save_session_vals,
                          es_save_extra_vals,
                          es_save_session_extra_vals)
from split_utils import (get_sbj_runs_ids,
                         index_data)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s", "--subject", type=int,
    help="Id of the subject to be processed",
    required=True, dest="subject_nr",
    choices=HOFF_SUBJECTS
)

parser.add_argument(
    "-id", "--input-dir-path", type=str,
    help="Path to the directory containing the data",
    required=True, dest="in_dir_path"
)

parser.add_argument(
    "-o", "--output-path", type=str,
    help="File in which we will save the data.",
    required=True, dest="out_path"
)

parser.add_argument(
    "-sh", "--shrinkage", type=float,
    help="LDA shrinkage hyper-parameter, Ledoit-Wolf lemma as default",
    required=False, dest="shrinkage",
    default=-1
)

parser.add_argument(
    "-sc", "--scaler", type=str,
    help="Scaler to apply to the data",
    required=True, dest="scaler",
    choices=["Standardize", "MinMax", "Original"]
)

parser.add_argument(
    "-tmax", "--maximum-trial", type=int,
    help="Number of trials required by the baseline method.",
    required=True, dest="tmax",
)

parser.add_argument(
    "-a", "--significance-threshold", type=float,
    help="Significance level from which we consider a p-value significant",
    required=False, dest="alpha",
    default=.05
)

parser.add_argument(
    "-st", "--statistical-test",
    help="Statistical test to employ",
    required=True, dest="stat",
    choices=list(map(lambda x: x.__name__, AVAILABLE_TESTS))
)

parser.add_argument(
    "-b", "--bonferroni",
    help="Whether to apply the Bonferroni correction or not",
    action="store_true",
    dest="bonferroni",
    default=False
)

if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    shrinkage = "auto" if not (
        0 <= args.shrinkage <= 1) else args.shrinkage
    for stat_curr in AVAILABLE_TESTS:
        if stat_curr.__name__ == args.stat:
            stat = stat_curr

    # Load data & reshape
    X, y, stim_seq, _, old_shape = load_data(
        args.in_dir_path, args.subject_nr)
    n_electr = old_shape[1]

    # Fixed runs' identifiers!
    runs_ids = get_sbj_runs_ids(
        len(HOFF_SESSIONS), len(HOFF_RUNS))

    # Splitting strategy
    logo = LeaveOneGroupOut()

    # Results
    results = dict()
    gain_vals = list()
    loss_vals = list()
    bar_t_vals = list()

    # Extra values
    bar_chosen_val_vals = list()
    std_chosen_val_vals = list()

    bar_ref_val_vals = list()
    std_ref_val_vals = list()

    # CV
    for train_index, test_index in tqdm(logo.split(X, y, runs_ids)):
        # Step 0: Train-Test-Indexing
        session_idx = runs_ids[test_index][0]
        X_train, y_train, stim_seq_train = index_data(
            train_index, X, y, stim_seq)
        X_test, y_test, stim_seq_test = index_data(
            test_index, X, y, stim_seq)

        # Step 1: Obtaining "t" and predicted sequence "z_soft"
        # Reshaping from
        # · X: (n_runs, n_epochs, n_feats) -> (n_runs x n_epochs, n_feats)
        # · y: (n_runs, n_epochs) -> (n_runs x n_epochs)
        X_train, y_train = sklearn_reshape(
            X_train, y_train)
        X_test, _ = sklearn_reshape(
            X_test, y_test)

        # Preprocess the data
        scaler = choose_scaler(
            n_electr, args.scaler)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Define the model
        clf = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage=shrinkage)

        # Training
        clf.fit(X_train, y_train)

        # Testing
        # Probability to be target
        y_pred = clf.predict_proba(X_test)[:, 1]

        # Soft-Voting
        y_pred, y_test, stim_seq_test = soft_voting_reshape(
            y_pred, y_test, stim_seq_test)

        y_pred = y_pred[:, :args.tmax]
        y_test = y_test[:, :args.tmax]
        stim_seq_test = stim_seq_test[:, :args.tmax]
        # Sort the stimulus & targets to compare
        y_pred = sort_stim(
            y_pred, stim_seq_test)
        y_diff = avg_diffs(y_pred)
        y_soft = soft_voting(
            y_pred, mean_div=False)
        y_test = sort_stim(
            y_test, stim_seq_test)

        # Step 2: Perform the statistical test
        p_vals = stat_test(
            y_diff, stat,
            alternative="two-sided"
        )

        # Obtain trial at which we stop per run
        t = stat_test_chosen_stim_es_trials(
            p_vals, y_soft,
            args.alpha, args.bonferroni
        )

        # Step 3: Obtain all the predictions
        z_soft = np.argmax(y_soft, axis=2)
        z_test = np.argmax(y_test, axis=2)

        run_idxs = np.arange(z_soft.shape[0])
        z_t = z_soft[run_idxs, t]

        # Obtain the accuracy per trial
        acc_base = np.count_nonzero(
            z_soft == z_test, axis=0)/len(run_idxs)
        acc_star = np.count_nonzero(
            z_t == z_test[run_idxs, t], axis=0)/len(run_idxs)

        # Step 4: Metrics
        score_ref = acc_base[-1]
        score_cur = acc_star

        bar_t = t.mean() + 1  # +1 because it is an index
        gain_curr = gain(
            t_max=args.tmax,
            t_star=bar_t
        )
        loss_curr = loss(
            acc_base=score_ref,
            acc_star=score_cur
        )

        bar_t_vals.append(bar_t)
        gain_vals.append(gain_curr)
        loss_vals.append(loss_curr)

        # Extra values
        es_extra_vals(
            bar_chosen_val_vals, std_chosen_val_vals,
            bar_ref_val_vals, std_ref_val_vals,
            y_soft, z_soft, z_test,
            t, args.tmax - 1  # -1 because it is not an index!
        )

        # Save results
        es_save_session_vals(
            results, session_idx, bar_t,
            args.alpha, gain_curr, loss_curr,
            score_ref, score_cur
        )

        # Extra results
        es_save_session_extra_vals(
            results, session_idx,
            bar_chosen_val_vals, std_chosen_val_vals,
            bar_ref_val_vals, std_ref_val_vals,
        )

    # Steps 5 & 6: Average along all the sessions left-out.
    es_save_vals(
        results,
        "Acc Avg Diff Stat Test - Fixed alpha:" +
        f"{args.alpha} - {args.stat} - " +
        f"Bonferroni: {args.bonferroni}",
        "alpha", np.mean(bar_t_vals),
        args.alpha, gain_vals, loss_vals
    )

    # Extra values
    es_save_extra_vals(
        results, bar_chosen_val_vals,
        bar_ref_val_vals
    )

    # Save the results
    os.makedirs(os.path.dirname(
        args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pkl.dump(results, f)
