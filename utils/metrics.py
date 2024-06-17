'''
This module will contain multiple
metrics such as classifier metrics,
BCI metrics, ...

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 04/02/2024
'''

import math
import numpy as np


def itr(n_classes: int,
        clf_acc: float,
        trial_secs: float) -> float:
    '''
    This method will compute the Information
    Transfer Rate of a classifier from its
    accuracy, number of choices, and prediction
    time of one trial.

    Example:
    ```
    itr(n_classes=6, clf_acc=.5, trial_secs=2)
    ----- Output -----
    12.719953598324249
    ```

    :param n_classes: Classifier's number of choices,
    or number of the problem's classes.
    :param clf_acc: Classifier's accuracy.
    :param trial_secs: One trial's prediction
    time in seconds.

    :return float: Information Transfer Rate metric
    in bits/min.
    '''
    if clf_acc >= 1:
        clf_acc = .999

    return (60/trial_secs) * (
        math.log2(n_classes) +
        clf_acc * math.log2(clf_acc) + (1 - clf_acc) *
        math.log2((1 - clf_acc)/(n_classes - 1)))


def bci_utility(
        n_classes: int,
        clf_acc: float,
        trial_secs: float) -> float:
    '''
    This method will compute the BCI-Utility
    metric of a classifier from its accuracy,
    number of choices, and prediction time of
    one trial.

    Example:
    ```
    bci_utility(n_classes=6, clf_acc=.6, trial_secs=2)
    ----- Output -----
    0.23219280948873616
    ```

    :param n_classes: Classifier's number of choices,
    or number of the problem's classes.
    :param clf_acc: Classifier's accuracy.
    :param trial_secs: One trial's prediction
    time in seconds.

    :return float: BCI-Utility metric.
    '''
    if clf_acc <= 0.5:
        return 0
    else:
        return (
            (2*clf_acc - 1) *
            (math.log2(n_classes - 1))
        )/trial_secs


def spm(
        n_classes: int,
        clf_acc: float,
        trial_secs: float) -> float:
    '''
    This method will compute the Symbols
    Per Minute of a classifier from its
    accuracy, number of choices, and prediction
    time of one trial.

    WARNING: The number of classes parameter is
    not used, this metric will correspond with
    a control interface instead of a speller,
    i.e. it can only make a decission within the
    `trial_secs` range. In other words, the number
    of decissions per minute is `(1/trial_secs)*(60s/1min)`.

    Example:
    ```
    spm(n_classes=6, clf_acc=.6, trial_secs=2)
    ----- Output -----
    5.999999999999998
    ```

    :param n_classes: Classifier's number of choices,
    or number of the problem's classes.
    :param clf_acc: Classifier's accuracy.
    :param trial_secs: One trial's prediction
    time in seconds.

    :return float: Symbols per minute.
    '''
    correct_percent = clf_acc
    incorrect_percent = 1 - clf_acc
    decs_per_min = (1/trial_secs) * 60  # From secs to mins

    return decs_per_min * (
        correct_percent - incorrect_percent)


def get_seq_pred(
        y_pred: np.ndarray,
        _max: bool) -> np.ndarray:
    '''
    This method will extract the sequence predicted
    by a classifier's prediction output.

    Example:
    ```
    np.random.seed(1234)

    n_runs = 2
    n_trials = 5
    n_flashes = 3

    X = np.random.randn(
        n_runs, n_trials, n_flashes)

    print("Values:")
    print(X)
    print("Sequence predicted:")
    print(get_seq_pred(X, _max=False))
    ----- Output -----
    Values:
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
    Sequence predicted:
    [[1 1 1 0 1]
     [0 1 0 1 0]]
    ```

    :param y_pred: 3D array with the predictions and
    shape: (n_runs, n_trials, n_flashes).
    :param _max: Boolean indicating whether to pick
    the highest value as the prediction or the minimum.

    :return np.ndarray: A 2D array with the predicted
    stimuli index per run & trial, i.e. with shape:
    (n_runs, n_trials).
    '''
    n_runs, n_trials, _ = y_pred.shape
    return np.argmax(y_pred, axis=2) if _max else\
        np.argmin(y_pred, axis=2)


def gain(t_max: int, t_star: int) -> float:
    '''
    This method will compute the gain between
    a fixed value and an optimised one.

    Example:
    ```
    print("GAIN:")
    print(gain(20, 10))
    ----- Output -----
    GAIN:
    0.5
    ```

    :param t_max: Fixed value from which obtain
    the gain.
    :param t_star: The optimised value.

    :return float: The metric.
    '''
    return (t_max - t_star)/t_max


def loss(acc_base: float, acc_star: float) -> float:
    '''
    This method will compute the loss between
    a fixed accuracy and another one.

    Example:
    ```
    print("LOSS:")
    print(loss(.8, .6))
    ----- Output -----
    LOSS:
    0.25000000000000006
    ```

    :param acc_base: Fixed value from which obtain
    the loss.
    :param acc_star: The other accuracy.

    :return float: The metric.
    '''
    return (acc_base - acc_star)/acc_base


def signed_r2_values(
        x: np.array,
        y: np.array,
        epochs_axis: int = 2) -> float:
    '''
    This method will compute the signed-r2-values.

    :param x: 3D-numpy array with all the data,
    the expected shape is:
        (n_feats, n_timesteps, n_epochs)
    :param y: Numpy array with all the labels,
    the expected shape is: (n_epochs,).
    :param epoch_axis: Axis along which the epochs
    will be indexed on `x` array, defaults to `2`.

    :return float: signed-r2-value with dimensions:
    (n_feat, n_timesteps).
    '''
    y_1_idxs = np.where(y == 1)[0]
    y_2_idxs = np.where(y == -1)[0]

    N_1 = len(y_1_idxs)
    N_2 = len(y_2_idxs)
    frac = np.sqrt(N_1 * N_2)/(N_1 + N_2)

    x_1 = np.take(
        x, y_1_idxs, axis=epochs_axis)
    x_2 = np.take(
        x, y_2_idxs, axis=epochs_axis)

    mu_1 = x_1.mean(axis=epochs_axis)
    mu_2 = x_2.mean(axis=epochs_axis)

    mu = x.mean(axis=epochs_axis)
    std = x.std(axis=epochs_axis)

    r = frac * (mu_1 - mu_2)/std
    sgn_r2_val = np.sign(mu) * (r * r)

    return sgn_r2_val
