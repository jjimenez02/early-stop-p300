'''
This module will define some utilities
to perform voting schemes such as soft-voting.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 15/02/2024
'''
import numpy as np
from tqdm import tqdm
from typing import Tuple
from metrics import get_seq_pred
from constants import HOFF_TRIAL_EPOCHS
from plots import (stim_power_bar_plot,
                   plot_accuracy_by_trial,
                   save_plot)


def soft_voting_reshape(
        X: np.array,
        y: np.array,
        stim_seq: np.array,
        n_flashes: int = HOFF_TRIAL_EPOCHS) -> Tuple[np.array,
                                                     np.array,
                                                     np.array]:
    '''
    Reshapes the input data into a format suitable
    for performing Soft-Voting.

    This function will add n_runs, n_trials, and
    n_flashes dimensions.

    Example:
    ```
    np.random.seed(1234)

    # Define the input data and target values
    X_train = np.random.rand(6*20*6)
    y_train = np.random.rand(6, 120)
    stim_seq = np.random.rand(6, 120)

    # Call the function to reshape the data
    new_X_train, new_y_train, new_stim_seq = soft_voting_reshape(
        X_train, y_train, stim_seq)

    new_X_train.shape, new_y_train.shape, new_stim_seq.shape
    ----- Output -----
    ((6, 20, 6), (6, 20, 6), (6, 20, 6))
    ```

    :param X: Input data of shape:
        (n_outputs)
    :param y: Target values of shape:
        (n_runs, n_epochs)
    :param stim_seq: Stimulus sequence with
    shape:
        (n_runs, n_epochs)
    :param n_flashes: Number of flashes per trial,
    default to `HOFF_TRIAL_EPOCHS`.

    :return Tuple[np.array x3]: Reshaped input
    data, target, and stimulus sequence values.
    All the arrays will have the following
    shape: `(n_runs, n_trials, n_flashes)`.
    '''
    n_runs, _ = y.shape

    X = X.reshape(
        n_runs,
        X.shape[0]//n_runs
    )
    X = X.reshape(
        X.shape[0],
        X.shape[1]//n_flashes,
        n_flashes
    )

    y = y.reshape(
        y.shape[0],
        y.shape[1]//n_flashes,
        n_flashes
    )

    stim_seq = stim_seq.reshape(
        stim_seq.shape[0],
        stim_seq.shape[1]//n_flashes,
        n_flashes
    )

    return X, y, stim_seq


def sort_stim(
        X: np.ndarray,
        stim_seq: np.ndarray) -> np.ndarray:
    '''
    This method will sort an array of values
    given a specific sequence.

    Example:
    ```
    np.random.seed(1234)

    n_runs = 1
    n_trials = 1
    n_flashes = 6

    seq_order = np.arange(n_flashes)
    np.random.shuffle(seq_order)

    X = np.random.randn(
        n_runs, n_trials, n_flashes)
    stim_seq = np.tile(
        np.tile(seq_order, n_trials),
        n_runs
    ).reshape((n_runs, n_trials, n_flashes))

    print("SORTED:")
    print(sort_stim(X, stim_seq))
    print("NOT SORTED:")
    print(X)
    ----- Output -----
    SORTED:
    [[[-0.6365235   0.88716294 -0.72058873 -2.24268495  0.01569637
        0.85958841]]]
    SEQUENCE:
    [2 1 5 0 4 3]
    NOT SORTED:
    [[[-0.72058873  0.88716294  0.85958841 -0.6365235   0.01569637
       -2.24268495]]]
    ```

    :param X: 3D Numpy array with shape:
        (n_runs, n_trials, n_flashes)
    :param stim_seq: Sequence of stimulus with shape:
        (n_runs, n_trials, n_flashes)
    :return np.ndarray: Accumulated predictions
    with shape `(n_runs, n_trials, n_flashes)`.
    '''
    # Re-arrange in function of stimulus presentation
    X_sorted = np.zeros_like(X)
    for run_idx in range(X.shape[0]):
        for trial_idx in range(X.shape[1]):
            idx = (run_idx, trial_idx)
            X_sorted[run_idx, trial_idx, stim_seq[idx]] =\
                X[run_idx, trial_idx]

    return X_sorted


def soft_voting(
        X: np.array,
        stim_seq: np.array = None,
        mean_div: bool = True) -> np.array:
    '''
    This method will compute the soft-voting
    weights of every flash.

    Example:
    ```
    np.random.seed(1234)

    # Define the input data and target values
    X_train = np.random.rand(6*20*6)
    y_train = np.random.rand(6, 120)
    stim_seq = np.tile(np.arange(
        0, 6), 120).reshape(6, 120)

    # Call the function to reshape the data
    new_X_train, new_y_train, new_stim_seq = soft_voting_reshape(
        X_train, y_train, stim_seq)

    soft_voting(new_X_train, new_stim_seq)[0, :3],\
        new_X_train[0, :3], new_stim_seq[0, :3]
    ----- Output -----
    (array([[0.19151945, 0.62210877, 0.43772774, 0.78535858, 0.77997581,
             0.27259261],
            [0.23399185, 0.71199047, 0.69793355, 0.83064561, 0.56889654,
             0.38679387],
            [0.38381555, 0.71222766, 0.58870595, 0.74082913, 0.54695875,
             0.26245206]]),
     array([[0.19151945, 0.62210877, 0.43772774, 0.78535858, 0.77997581,
             0.27259261],
            [0.27646426, 0.80187218, 0.95813935, 0.87593263, 0.35781727,
             0.50099513],
            [0.68346294, 0.71270203, 0.37025075, 0.56119619, 0.50308317,
             0.01376845]]),
     array([[0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]]))
    ```

    :param X: Predictions with shape:
        (n_runs, n_trials, n_flashes)
    :param stim_seq: Sequence of stimulus with shape:
        (n_runs, n_trials, n_flashes)
    If its value is `None`, it won't sort the array
    and therefore it could be summed wrongly! Default
    to `None`.
    :param mean_div: Whether to divide every result by
    the number of trials seen or not, default to True.
    :return np.array: Accumulated predictions
    with shape `(n_runs, n_trials, n_flashes)`.
    '''
    if stim_seq is not None:
        X_soft = sort_stim(X, stim_seq)
    else:
        X_soft = np.copy(X)

    # Soft-voting
    cum_res = np.cumsum(X_soft, axis=1)
    if mean_div:
        return cum_res/np.arange(
            1, X_soft.shape[1] + 1)[None, :, None]
    else:
        return cum_res


def avg_diffs(
        X: np.ndarray,
        stim_seq: np.ndarray = None,
        mean_div: bool = False) -> np.ndarray:
    '''
    This method will obtain the averaged differences
    of one stimuli against the rest.

    Example:
    ```
    np.random.seed(1234)

    n_runs = 2
    n_trials = 3
    n_flashes = 3

    # Define the input data and target values
    X = np.random.rand(
        n_runs, n_trials, n_flashes)
    stim_seq = np.tile(np.arange(
        0, n_flashes), n_trials*n_runs).reshape(
            (n_runs, n_trials, n_flashes))

    # Call the function to reshape the data
    X_diff = avg_diffs(X, stim_seq)

    print("Original:")
    print(X[0, 0])
    print("DIFFS:")
    print(X_diff[0, 0])
    ----- Output -----
    Original:
    [0.19151945 0.62210877 0.43772774]
    DIFFS:
    [ 0.3383988  -0.30748518 -0.03091363]
    ```

    :param X: Predictions with shape:
        (n_runs, n_trials, n_flashes)
    :param stim_seq: Sequence of stimulus with shape:
        (n_runs, n_trials, n_flashes)
    If its value is `None`, it won't sort the array
    and therefore it could be differenced wrongly! Default
    to `None`.
    :param mean_div: Whether to divide every result by
    the number of trials seen or not, default to False.
    :return np.ndarray: A new numpy array with the
    averaged differences per run, trial and stimuli, i.e.
    with shape:
        (n_runs, n_trials, n_flashes)
    '''
    if stim_seq is not None:
        X_diff = sort_stim(X, stim_seq)
    else:
        X_diff = np.copy(X)

    _, n_trials, n_flashes = X.shape
    for stim_idx in range(n_flashes):
        diffs = X[:, :, [stim_idx]] - X
        # -1 to avoid considering the same flash
        X_diff[:, :, stim_idx] =\
            np.sum(diffs, axis=2) / (n_flashes - 1)

    if mean_div:
        return X_diff/(np.arange(n_trials) + 1).reshape((1, -1, 1))
    else:
        return X_diff


def all_sv_plots(
        fig_path: str,
        X_soft: np.array,
        stim_tgt_test: np.array,
        metric: callable):
    '''
    This method will plot all the Soft-Voting
    figures.

    Example:
    ```
    from sklearn.metrics import accuracy_score

    all_sv_plots(
        "lmao/hello",
        np.random.randn(6, 20, 6),
        np.random.randint(0, 5+1, size=6),
        accuracy_score
    )
    ```

    :param fig_path: Directory path in which we
    will save the figures.
    :param X_soft: Array of soft-voted predictions
    with shape: (n_runs, n_trials, n_flashes).
    :param stim_tgt_test: Array of ground-truths
    with shape: (n_runs,).
    :param metric: Metric from which we will compute
    the soft-voting score plot, it should have an `y_test`,
    and `y_pred` parameter, and a `__name__`.
    '''
    n_runs, n_trials, _ = X_soft.shape

    # Make stimuli power-bar-plot for every trial
    stims = np.arange(HOFF_TRIAL_EPOCHS)
    stims_labels = [f"Stim {stim}" for stim in stims]
    for run_idx in tqdm(range(n_runs)):
        run_values = X_soft[run_idx]

        labels = [
            f"{label} - Anomalous: " + str(stim == stim_tgt_test[run_idx])
            for stim, label in zip(stims, stims_labels)
        ]

        ymin = np.min(run_values)
        ymax = np.max(run_values)
        for trial_idx in range(n_trials):
            stim_power_bar_plot(
                stims, run_values[trial_idx], labels,
                "Stimulus", "Accumulated prediction",
                ymin, ymax, title=f"Trial nr: {trial_idx + 1}"
            )
            save_plot(fig_path + f"{run_idx}/" + f"{trial_idx}")

    # Make scoring plot
    stim_tgt_pred = get_seq_pred(X_soft, _max=True)

    scores = list()
    for trial_idx in range(n_trials):
        scores.append(metric(
            stim_tgt_test,
            stim_tgt_pred[:, trial_idx]
        ))

    plot_accuracy_by_trial(
        scores, metric.__name__)
    save_plot(fig_path + "Trial evolution")
