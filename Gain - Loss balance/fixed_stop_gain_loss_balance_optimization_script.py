'''
This script will obtain the optimised
trial balancing the gain & loss metrics
(taking the intersection of both curves)
leaving every session out for a single
subject.

This method does not consider any.
inter-subject feature.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 16/04/2024
'''

import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from metrics import (gain,
                     loss)
from constants import (SEED,
                       HOFF_SUBJECTS,
                       HOFF_SESSIONS,
                       HOFF_RUNS)
from data_utils import (load_data,
                        sklearn_reshape)
from transformers import choose_scaler
from voting_utils import (soft_voting,
                          soft_voting_reshape,
                          sort_stim)
from report_utils import (es_extra_vals,
                          es_save_vals,
                          es_save_session_vals,
                          es_save_extra_vals,
                          es_save_session_extra_vals)
from split_utils import (get_sbj_runs_ids,
                         index_data)
from gain_loss import obtain_gain_loss_shared_max
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
    "-od", "--output-dir-path", type=str,
    help="Path to the directory in which we'll save the data",
    required=True, dest="out_dir_path"
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

if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    shrinkage = "auto" if not (
        0 <= args.shrinkage <= 1) else args.shrinkage

    # Output dir
    out_dir = Path(args.out_dir_path)
    os.makedirs(os.path.dirname(
        out_dir), exist_ok=True)

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

        # 3-Fold CV
        runs_ids_cv = get_sbj_runs_ids(
            # -1 because we left one session out
            len(HOFF_SESSIONS) - 1, len(HOFF_RUNS))
        t_vals = list()
        for fold_idx, (cv_train_index, cv_val_index) in enumerate(logo.split(
                X_train, y_train, runs_ids_cv)):
            # Step 1: Validation indexing
            X_train_cv, y_train_cv, stim_seq_train_cv = index_data(
                cv_train_index, X_train, y_train, stim_seq_train)
            X_val_cv, y_val_cv, stim_seq_val_cv = index_data(
                cv_val_index, X_train, y_train, stim_seq_train)

            # Reshaping from
            # · X: (n_runs, n_epochs, n_feats) -> (n_runs x n_epochs, n_feats)
            # · y: (n_runs, n_epochs) -> (n_runs x n_epochs)
            X_train_cv, y_train_cv = sklearn_reshape(
                X_train_cv, y_train_cv)
            X_val_cv, _ = sklearn_reshape(
                X_val_cv, y_val_cv)

            # Preprocess the data
            scaler = choose_scaler(
                n_electr, args.scaler)
            scaler.fit(X_train_cv)
            X_train_cv = scaler.transform(X_train_cv)
            X_val_cv = scaler.transform(X_val_cv)

            # Define the model
            clf = LinearDiscriminantAnalysis(
                solver="eigen", shrinkage=shrinkage)

            # Training
            clf.fit(X_train_cv, y_train_cv)

            # Step 2: Validation
            # Probability to be target
            y_pred_cv = clf.predict_proba(X_val_cv)[:, 1]

            # Steps 3 & 4: Soft-Voting
            y_pred_cv, y_val_cv, stim_seq_val_cv = soft_voting_reshape(
                y_pred_cv, y_val_cv, stim_seq_val_cv)

            y_pred_cv = y_pred_cv[:, :args.tmax]
            y_val_cv = y_val_cv[:, :args.tmax]
            stim_seq_val_cv = stim_seq_val_cv[:, :args.tmax]
            # Sort the stimulus to perform Soft-Voting
            y_pred_cv = sort_stim(
                y_pred_cv, stim_seq_val_cv)
            y_soft_cv = soft_voting(
                y_pred_cv, mean_div=False)

            # Sort the target as well to compare
            y_val_cv = sort_stim(
                y_val_cv, stim_seq_val_cv)

            # Step 5: Obtain the stimuli predicted per trial
            z_soft_cv = np.argmax(y_soft_cv, axis=2)
            z_val_cv = np.argmax(y_val_cv, axis=2)

            # Step 6: Obtain the accuracy per trial
            scores_t = np.count_nonzero(
                # -1 to turn it into an index
                z_soft_cv == z_val_cv,
                axis=0
            )/z_soft_cv.shape[0]

            # Step 7: Obtain the loss per trial
            score_base = scores_t[-1]
            loss_t = loss(score_base, scores_t)

            # Step 8: Obtain the gain per trial
            trials = np.arange(1, args.tmax + 1)
            gain_t = gain(args.tmax, trials)

            # Step 9: Obtain the intersection
            fst_coincidence = np.argmax(
                z_soft_cv == z_val_cv,
                axis=1
            ).mean() + 1  # +1 because it is an index

            t = obtain_gain_loss_shared_max(
                gain_t, loss_t, trials,
                [1, args.tmax],
                round=True, plot=True,
                out_dir=out_dir,
                title=f"Session {session_idx} Out - " +
                f"Fold {fold_idx}",
                xlabel="Trial",
                xtick_labels=trials
            )

            t_vals.append(t)

        # Step 10: Average along all the folds
        bar_t = np.round(np.mean(t_vals))
        bar_t_vals.append(bar_t)

        # Step 11: Obtaining GAIN & LOSS metrics on test
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
        # Sort the stimulus to perform Soft-Voting
        y_pred = sort_stim(
            y_pred, stim_seq_test)
        y_soft = soft_voting(
            y_pred, mean_div=False)

        # Sort the target as well to compare
        y_test = sort_stim(
            y_test, stim_seq_test)

        # Obtain the stimuli predicted per trial
        z_soft = np.argmax(y_soft, axis=2)
        z_test = np.argmax(y_test, axis=2)

        # Obtain the accuracy per trial
        scores_test = np.count_nonzero(
            z_soft == z_test, axis=0)/z_soft.shape[0]

        # Metrics
        score_ref = scores_test[-1]
        # -1 because it is not an index!
        score_cur = scores_test[int(bar_t) - 1]

        gain_curr = gain(
            t_max=args.tmax,
            t_star=bar_t
        )
        loss_curr = loss(
            acc_base=score_ref,
            acc_star=score_cur
        )

        gain_vals.append(gain_curr)
        loss_vals.append(loss_curr)

        # Extra values
        t = int(bar_t) - 1  # -1 because it is not an index!
        es_extra_vals(
            bar_chosen_val_vals, std_chosen_val_vals,
            bar_ref_val_vals, std_ref_val_vals,
            y_soft, z_soft, z_test,
            t, args.tmax - 1  # -1 because it is not an index!
        )

        # Save results
        es_save_session_vals(
            results, session_idx, bar_t,
            bar_t, gain_curr, loss_curr,
            score_ref, score_cur
        )

        # Extra results
        es_save_session_extra_vals(
            results, session_idx,
            bar_chosen_val_vals, std_chosen_val_vals,
            bar_ref_val_vals, std_ref_val_vals,
        )

    # Steps 12 & 13: Average along all the sessions left-out.
    bar_bar_t = np.mean(bar_t_vals)
    es_save_vals(
        results,
        "Fixed Stop - G-L Bal",
        "trial", bar_bar_t,
        bar_bar_t, gain_vals,
        loss_vals
    )

    # Extra values
    es_save_extra_vals(
        results, bar_chosen_val_vals,
        bar_ref_val_vals
    )

    # Save the results
    with open(out_dir / "results.pkl", "wb") as f:
        pkl.dump(results, f)
