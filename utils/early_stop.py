'''
This module will define some utilities
to perform multiple early stop criterias
over an array with predictions and shape:
    (n_runs, n_trials, n_flashes)

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 06/04/2024
'''

import numpy as np
from typing import Tuple


def fixed_stop(
        y_pred: np.ndarray,
        trials_idxs: np.ndarray) -> np.ndarray:
    '''
    This method will apply the fixed-stop
    criteria by fixing the predictions
    posterior to a specific series of trial
    indexes per run.

    Example:
    ```
    np.random.seed(1234)

    n_runs = 2
    n_trials = 5
    n_flashes = 3

    X = np.random.randn(
        n_runs, n_trials, n_flashes)
    trials_idxs = np.repeat(2, n_runs)

    X_fixed = fixed_stop(X, trials_idxs)

    print("Original:")
    print(X)
    print("Fixed:")
    print(X_fixed)
    ----- Output -----

    Original:
    [[[ 4.71435164e-01 -1.19097569e+00  1.43270697e+00]
      [-3.12651896e-01 -7.20588733e-01  8.87162940e-01]
      [ 8.59588414e-01 -6.36523504e-01  1.56963721e-02]
      [-2.24268495e+00  1.15003572e+00  9.91946022e-01]
      [ 9.53324128e-01 -2.02125482e+00 -3.34077366e-01]]

     [[ 2.11836468e-03  4.05453412e-01  2.89091941e-01]
      [ 1.32115819e+00 -1.54690555e+00 -2.02646325e-01]
      [-6.55969344e-01  1.93421376e-01  5.53438911e-01]
      [ 1.31815155e+00 -4.69305285e-01  6.75554085e-01]
      [-1.81702723e+00 -1.83108540e-01  1.05896919e+00]]]
    Fixed:
    [[[ 0.47143516 -1.19097569  1.43270697]
      [-0.3126519  -0.72058873  0.88716294]
      [ 0.85958841 -0.6365235   0.01569637]
      [ 0.85958841 -0.6365235   0.01569637]
      [ 0.85958841 -0.6365235   0.01569637]]

     [[ 0.00211836  0.40545341  0.28909194]
      [ 1.32115819 -1.54690555 -0.20264632]
      [-0.65596934  0.19342138  0.55343891]
      [-0.65596934  0.19342138  0.55343891]
      [-0.65596934  0.19342138  0.55343891]]]
    ```

    :param y_pred: Array with the predictions
    and shape: (n_runs, n_trials, n_flashes)
    :param trials_idxs: A 1D array with the trials'
    indexes per run from which we will fix our
    predictions.

    :return np.ndarray: An array with the same
    shape as `y_pred`, but with the fixed-stop
    criteria applied.
    '''
    n_runs, n_trials, _ = y_pred.shape
    y_es = np.zeros_like(y_pred)

    for run_idx, trial_idx in enumerate(trials_idxs):
        if trial_idx < n_trials:
            trial_pred = y_pred[run_idx, trial_idx].reshape(
                1, -1)
            y_es[run_idx] += trial_pred
            y_es[run_idx, :trial_idx] =\
                y_pred[run_idx, :trial_idx]

    return y_es


def evid_threshold_stop(
        y_pred: np.ndarray,
        threshold: float) -> Tuple[np.ndarray,
                                   np.ndarray]:
    '''
    This method will return the predicted
    sequence per run and the trial at which
    they were predicted (per run as well) given
    a specific evidence threshold.

    WARNING: Note that if there is more than one
    stimuli above the specified threshold, we will
    pick up the highest one.

    Also note that this function expects `y_pred` to
    be sorted by stimuli, i.e. y_pred[x, y, 0] will
    always correspond to the first stimuli and not
    any other one (see `voting_utils.sort_stim`).

    Example:
    ```
    np.random.seed(1234)

    n_runs = 2
    n_trials = 5
    n_flashes = 3

    X = np.random.randn(
        n_runs, n_trials, n_flashes)

    t, z_pred = evid_threshold_stop(
        X, threshold=1)

    print("Original:")
    print(X)
    print("Trials:")
    print(t)
    print("Stims predicted:")
    print(z_pred)

    ----- Output -----
    Original:
    [[[ 4.71435164e-01 -1.19097569e+00  1.43270697e+00]
      [-3.12651896e-01 -7.20588733e-01  8.87162940e-01]
      [ 8.59588414e-01 -6.36523504e-01  1.56963721e-02]
      [-2.24268495e+00  1.15003572e+00  9.91946022e-01]
      [ 9.53324128e-01 -2.02125482e+00 -3.34077366e-01]]

     [[ 2.11836468e-03  4.05453412e-01  2.89091941e-01]
      [ 1.32115819e+00 -1.54690555e+00 -2.02646325e-01]
      [-6.55969344e-01  1.93421376e-01  5.53438911e-01]
      [ 1.31815155e+00 -4.69305285e-01  6.75554085e-01]
      [-1.81702723e+00 -1.83108540e-01  1.05896919e+00]]]
    Trials:
    [0 1]
    Stims predicted:
    [2 0]
    ```

    :param y_pred: Array with the predictions
    and shape: (n_runs, n_trials, n_flashes)
    :param threshold: Evidence threshold from
    which we will consider a stimuli the target.

    :return Tuple[np.ndarray x2]: Two 1D numpy arrays,
    the first one containing the trial index at which
    every run stopped, and the second one the predicted
    stimuli index per run, expected shapes for both arrays:
    (n_runs,)
    '''
    n_runs, n_trials, _ = y_pred.shape
    # -1 to turn them into indexes
    t = np.zeros(n_runs, dtype=int) + (n_trials - 1)

    # First trial reaching the threshold per run
    y_pred_th = np.any(y_pred >= threshold, axis=2)
    runs_idxs, trials_idxs = np.where(y_pred_th)

    # Unique indexes
    runs_idxs, first_occurrence_idxs = np.unique(
        runs_idxs, return_index=True)
    t[runs_idxs] = trials_idxs[first_occurrence_idxs]

    # Chosen stimuli per run & trial
    z_pred = np.argmax(y_pred, axis=2)

    idxs = np.arange(n_runs)
    return t, z_pred[idxs, t]


def stat_test(
        y_pred: np.ndarray,
        test: callable,
        **test_kwargs) -> np.ndarray:
    '''
    This method will compute the p-values between
    different stimulus predictions given by a
    classifier.

    WARNING: Note that the estimations with a single
    sample will be saved all with ones to avoid NaN
    values (this should happen just for the first trial).

    Example:
    ```
    np.random.seed(1234)

    n_runs = 1
    n_trials = 500
    n_flashes = 3

    X = norm.rvs(
        loc=5, scale=10,
        size=(n_runs, n_trials, n_flashes)
    )

    X[:, :, 0] = norm.rvs(
        loc=20, scale=10,
        size=(n_runs, n_trials)
    )

    p_vals = stat_test(
        X, ttest_ind, alternative="two-sided")

    print("RESULTS:")
    print(p_vals[0, -1])
    ----- Output -----
    RESULTS:
    [[1.00000000e+000 2.38184729e-111 1.35726577e-100]
     [2.38184729e-111 1.00000000e+000 6.29418499e-002]
     [1.35726577e-100 6.29418499e-002 1.00000000e+000]]
    ```

    :param y_pred: Array with the predictions
    and shape: (n_runs, n_trials, n_flashes)
    :param test: Method to use as statistical test, it
    will receive two 3D arrays as input with shapes:
        - (n_runs, n_samples, 1)
        - (n_runs, n_samples, n_flashes)
    And it should return an object with a `pvalue` attribute
    which should contain a 3D array with shape:
        (n_runs, n_samples, n_flashes)
    :param **test_kwargs: Extra arguments for the `test`
    function.

    :return np.ndarray: A numpy array with shape:
        (n_runs, n_trials, n_flashes, n_flashes)
    Whose elements will be the p-value between two
    specific stimulus within the same run and trial.
    '''
    n_runs, n_trials, n_flashes = y_pred.shape
    p_vals = np.ones((
        n_runs, n_trials,
        n_flashes, n_flashes
    ))

    # Note that `trial_nr` represents the trials
    # seen until now! We start at `2` because the `:2`
    # slice does not include the last element, i.e. we are
    # just seeing 0 & 1!
    for trial_nr in range(2, n_trials + 1):
        for stim_idx in range(n_flashes):
            trial_idx = trial_nr - 1  # In which trial we are at.

            curr_stim = y_pred[:, :trial_nr, stim_idx].reshape(
                (n_runs, -1, 1))

            p_vals[:, trial_idx, stim_idx] =\
                test(curr_stim, y_pred[:, :trial_nr],
                     axis=1, **test_kwargs).pvalue

    return p_vals


def stat_test_chosen_stim_es_trials(
        p_vals: np.ndarray,
        y_soft: np.ndarray,
        alpha: float,
        bonferroni: bool = True) -> np.ndarray:
    '''
    This method will obtain the trial indexes at
    which a statistical test + early stopping algorithm
    would have stopped per run for the current stimuli
    predicted.

    Example:
    ```
    np.random.seed(1234)

    n_runs = 3
    n_trials = 500
    n_flashes = 3

    X = norm.rvs(
        loc=5, scale=10,
        size=(n_runs, n_trials, n_flashes)
    )

    X[:, :, 0] = norm.rvs(
        loc=20, scale=10,
        size=(n_runs, n_trials)
    )

    p_vals = stat_test(
        X, ttest_ind, alternative="two-sided")

    run_es = stat_test_chosen_stim_es_trials(
        p_vals, X.cumsum(axis=1), alpha=.05)

    print("RESULTS:")
    print(run_es)
    ----- Output -----
    RESULTS:
    [11  4  2]
    ```

    :param p_vals: A 4D array with the p-values and
    shape: (n_runs, n_trials, n_flashes, n_flashes)
    :param y_soft: A 3D array with the accumulated
    predictions and shape: (n_runs, n_trials, n_flashes)
    :param alpha: Significance level.
    :param bonferroni: Whether to apply the Bonferroni
    correction or not, defaults to True.

    :return np.ndarray: It will return a 1D array with
    the trial index at which we stop per run.
    '''
    n_runs, n_trials, n_flashes, _ = p_vals.shape
    # -1 because we do not count the test
    # of the predicted stimuli with itself
    n_tests = n_flashes - 1

    # Predicted stimulus per trial and run
    y_tgt_pred = np.argmax(y_soft, axis=2)

    # Bonferroni correction
    alpha = alpha/n_tests if bonferroni else alpha

    # `... + n_trials` to consider the non-stopping scenario
    # and `-1` because they should be indexes
    run_es = (np.zeros(n_runs) + n_trials - 1).astype(int)
    for run_idx in range(n_runs):
        for trial_idx in range(n_trials):
            stim_idx = y_tgt_pred[run_idx, trial_idx]
            p_vals_sign_count = np.count_nonzero(
                p_vals[run_idx, trial_idx, stim_idx] < alpha)

            if p_vals_sign_count == n_tests:
                run_es[run_idx] = trial_idx
                break

    return run_es


def stat_test_all_stim_es_trials(
        p_vals: np.ndarray,
        y_pred: np.ndarray,
        alpha: float,
        bonferroni: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    This method will obtain the trial indexes at
    which a statistical test + early stopping algorithm
    would have stopped per run (and the predicted stimuli's
    index).

    WARNING: Here we will consider that test(X1, X2) = test(X2, X1)!

    Example:
    ```
    np.random.seed(1234)

    n_runs = 3
    n_trials = 500
    n_flashes = 3

    X = norm.rvs(
        loc=5, scale=10,
        size=(n_runs, n_trials, n_flashes)
    )

    X[:, :, 0] = norm.rvs(
        loc=20, scale=10,
        size=(n_runs, n_trials)
    )

    p_vals = stat_test(
        X, ttest_ind, alternative="two-sided")

    y_pred = np.random.randint(
        0, 2, size=(n_runs, n_trials, n_flashes))
    run_es, run_pred = stat_test_all_stim_es_trials(
        p_vals, y_pred, alpha=.05)

    print("Stopped at:")
    print(run_es)
    print("Predictions:")
    print(run_pred)
    ----- Output -----
    Stopped at:
    [9 4 3]
    Predictions:
    [1 0 0]
    ```

    :param p_vals: A 4D array with the p-values and
    shape: (n_runs, n_trials, n_flashes, n_flashes)
    :param y_pred: A 3D array with the prediction
    values and shape: (n_runs, n_trials, n_flashes)
    :param alpha: Significance level.
    :param bonferroni: Whether to apply the Bonferroni
    correction or not, defaults to True.

    :return np.ndarray: It will return a 1D array with
    the trial index at which we stop per run.
    '''
    n_runs, n_trials, n_flashes = y_pred.shape
    # The number of combinations of paired tests without
    # repetition, we are assuming that test(X1, X2) = test(X2, X1)!
    n_tests = n_flashes * (n_flashes - 1) / 2

    # Bonferroni correction
    alpha = alpha/n_tests if bonferroni else alpha

    # Significant stimulus
    p_vals_sign_count = np.count_nonzero(
        p_vals < alpha, axis=3)
    # -1 to avoid including the same stimuli
    sign_cells = p_vals_sign_count == (n_flashes - 1)

    run_es = np.zeros(n_runs, dtype=int)
    run_pred = np.zeros(n_runs, dtype=int)
    for run_idx in range(n_runs):
        trials_n_sign_stims = np.count_nonzero(
            sign_cells[run_idx], axis=1)
        stop_idxs = np.argwhere(
            trials_n_sign_stims == 1).flatten()

        if len(stop_idxs) != 0:
            chosen_trial = stop_idxs[0]
            chosen_stim = np.argwhere(
                sign_cells[run_idx, chosen_trial].flatten()).flatten()
        else:
            chosen_trial = n_trials - 1  # -1 to obtain the index
            sign_stims = np.argwhere(
                sign_cells[run_idx, chosen_trial].flatten()).flatten()

            # If there is no significant stimuli, then all are significant now!
            if len(sign_stims) == 0:
                sign_stims = np.arange(n_flashes)

            p_vals_weights = 1 - np.sum(
                p_vals[run_idx, chosen_trial, sign_stims], axis=1)
            sign_stims_preds = y_pred[run_idx, chosen_trial, sign_stims]

            chosen_stim = sign_stims[np.argmax(
                p_vals_weights * sign_stims_preds)]

        run_es[run_idx] = chosen_trial
        run_pred[run_idx] = chosen_stim

    return run_es, run_pred
