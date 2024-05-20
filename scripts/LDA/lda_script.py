'''
This script will load all the runs of
a single subject, and apply the necessary
preprocessing to apply the Linear Discriminant
Analysis training.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 18/02/2024
'''


import os
import argparse
import numpy as np
import pickle as pkl
from data_utils import (load_data,
                        sklearn_reshape)
from split_utils import (get_logo_idxs,
                         get_sbj_runs_ids,
                         index_data)
from constants import (HOFF_SUBJECTS,
                       HOFF_SESSIONS,
                       HOFF_RUNS,
                       SEED)
from transformers import (SkLearnWrapper,
                          TimeSeriesMinMaxScaler,
                          TimeSeriesStandardScaler,
                          TimeSeriesIdentityScaler)
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
    help="Path in which we will save the model's outputs",
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


if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    shrinkage = "auto" if not (
        0 <= args.shrinkage <= 1) else args.shrinkage

    # Load data & reshape
    X, y, stim_seq, stim_tgt, old_shape = load_data(
        args.in_dir_path, args.subject_nr)
    n_electr = old_shape[1]

    # Fixed runs' identifiers!
    runs_ids = get_sbj_runs_ids(
        len(HOFF_SESSIONS), len(HOFF_RUNS))

    # Leave-One-Session-Out
    train_idxs, test_idxs = get_logo_idxs(runs_ids)
    X_train, y_train, stim_seq_train, stim_tgt_train = index_data(
        train_idxs, X, y, stim_seq, stim_tgt)
    X_test, y_test, stim_seq_test, stim_tgt_test = index_data(
        test_idxs, X, y, stim_seq, stim_tgt)

    # Training reshape (we don't need n_runs anymore)
    X_train, y_train = sklearn_reshape(X_train, y_train)
    X_test, _ = sklearn_reshape(X_test, y_test)

    # Define the model
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage=shrinkage)
    if args.scaler == "MinMax":
        scaler = SkLearnWrapper(
            n_electr,
            TimeSeriesMinMaxScaler(
                feature_range=(-1, 1))
        )
    elif args.scaler == "Standardize":
        scaler = SkLearnWrapper(
            n_electr,
            TimeSeriesStandardScaler()
        )
    else:
        scaler = SkLearnWrapper(
            n_electr,
            TimeSeriesIdentityScaler()
        )
    scaler.fit(X_train)

    # Extra Pre-processing
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model with the training data
    lda.fit(X_train, y_train)

    # Predict and save output
    # We pick up the probability of belonging to the target class
    y_pred = lda.predict_proba(X_test)[:, 1]

    lda_info = (
        [], [], [], [], [],
        [y_pred, y_test, stim_seq_test, stim_tgt_test]
    )

    os.makedirs(os.path.dirname(
        args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pkl.dump(lda_info, f)
