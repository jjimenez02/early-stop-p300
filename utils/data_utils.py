'''
This module will define some utilites
when loading the data.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 15/02/2024
'''

import numpy as np
import pickle as pkl
from pathlib import Path
from path_utils import get_run_path
from constants import (HOFF_SESSIONS,
                       HOFF_RUNS)
from typing import Tuple, List, Union


def load_data(
        dir_path: str,
        subject_nr: int,
        reshape: bool = True) -> Tuple[np.array, np.array,
                                       np.array, np.array,
                                       tuple]:
    '''
    This method will load all the subject
    runs from a specific path.

    Example:
    ```
    data_path = "preproc/dummy_AE"
    subject_nr = 1

    X, y, stim_seq, stim_tgt, old_shape = load_data(data_path, subject_nr)
    X.shape, y.shape, stim_seq.shape, stim_tgt.shape, old_shape
    ----- Output -----
    ((24, 120, 1024), (24, 120), (24, 120), (24,), (24, 32, 20, 6, 32))
    ```

    :param dir_path: Input's directory.
    :param subject_nr: Subject's identifier.
    :param reshape: Boolean to indicate whether we should
    apply a reshape into (n_runs, n_epochs, n_features)
    or load:
        (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
    :return Tuple[np.array x4, tuple]:

    Depending on `reshape` it will return:
    - If it is True:
        A tuple with (respectively)
        the run's data with shape:
            (n_runs, n_epochs, n_features)
        The targets with shape:
            (n_runs, n_epochs)
        The stimulus sequence with shape:
            (n_runs, n_epochs)
        And the stimulus targets with shape:
            (n_runs,)
        And the old shape of `X`:
            (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
    - If it is False:
        A tuple with (respectively)
        the run's data with shape:
            (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
        The targets with shape:
            (n_runs, n_trials, n_flashes)
        The stimulus sequence with shape:
            (n_runs, n_trials, n_flashes)
        And the stimulus targets with shape:
            (n_runs,)
        And the shape of `X`:
            (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
    '''
    X, y, stim_seq, stim_tgt =\
        list(), list(), list(), list()
    for session_nr in HOFF_SESSIONS:
        for run_nr in HOFF_RUNS:
            run_path = get_run_path(
                dir_path, subject_nr,
                session_nr, run_nr
            )

            with open(run_path, "rb") as f:
                X_run, y_run, stim_seq_run, stim_tgt_run =\
                    pkl.load(f)

            X.append(X_run)
            y.append(y_run)
            stim_seq.append(stim_seq_run),
            stim_tgt.append(stim_tgt_run)

    X = np.asarray(X)
    y = np.asarray(y)
    stim_seq = np.asarray(stim_seq)
    stim_tgt = np.asarray(stim_tgt)

    if reshape:
        return __data_reshape(X, y, stim_seq, stim_tgt)
    else:
        return X, y, stim_seq, stim_tgt, X.shape


def load_electrodes(
        dir_path: str,
        subject_nr: int) -> List[str]:
    '''
    This method will load all the electrodes'
    names preprocessed from a preprocessed subject
    directory.

    Example:
    ```
    data_path = "../AutoEncoder Outlier Detection/preproc/dummy_AE"
    subject_nr = 1

    load_electrodes(data_path, subject_nr)
    ----- Output -----
    array(['FP1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
           'P7', 'P3', 'PZ', 'PO3', 'O1', 'OZ', 'O2', 'PO4', 'P4', 'P8',
           'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2',
           'FZ', 'CZ'], dtype='<U3')
    ```

    :param dir_path: Input's directory.
    :param subject_nr: Subject's identifier.
    :return List[str]: A list with all the used
    electrodes.
    '''
    path = Path(dir_path) / f"subject{subject_nr}" /\
        "electrodes.pkl"
    with open(path, "rb") as f:
        elect_names = pkl.load(f)
    return elect_names


def sklearn_reshape(
        X: np.array,
        y: np.array = None) -> Union[Tuple[
            np.array, np.array], np.array]:
    """
    Reshapes the input data into a format suitable
    for training/predicting with a Scikit-Learn's
    machine learning model.

    This function will combine `n_runs` & `n_epochs`
    axes.

    Example:
    ```
    np.random.seed(1234)

    # Define the input data and target values
    X_train = np.random.rand(5, 6, 32)
    y_train = np.random.rand(5, 6)

    # Call the function to reshape the data
    new_X_train, new_y_train = training_reshape(
        X_train, y_train)

    new_X_train.shape, new_y_train.shape
    ----- Output -----
    ((30, 32), (30,))
    ```

    :param X: Input data of shape:
        (n_runs, n_epochs, n_feats)
    :param y: Target values of shape:
        (n_runs, n_epochs)

    :return Union[Tuple[np.array x2], np.array]: Reshaped input
    data and target values (only if it is not None). In particular:
    The values with shape:
        (n_runs x n_epochs, n_feats)
    The targets with shape:
        (n_runs x n_epochs)

    """
    n_runs, n_epochs, n_feats = X.shape

    # Reshape the input data into a format
    # suitable for SkLearn's models
    new_X = X.reshape(
        n_runs * n_epochs, n_feats)
    if y is not None:
        new_y = y.reshape(
            n_runs * n_epochs)

        return new_X, new_y
    else:
        return new_X


def gen_secs_spent_arr(
        epoch_len: float,
        isi: float,
        trial_epochs: int,
        n_trials: int,
        overlapped: bool) -> np.ndarray:
    '''
    This method will create an array with the already
    spent time in seconds per trial.

    Example:
    ```
    print(gen_secs_spent_arr(
        epoch_len=1, isi=.4,
        trial_epochs=6, n_trials=20,
        overlapped=True
    ))
    ----- Output -----
    [ 3.  6.  9. 12. 15. 18. 21. 24. 27. 30. 33. 36. 39. 42. 45. 48. 51. 54.
     57. 60.]
    ```

    :param epoch_len: Length in seconds of
    an epoch.
    :param isi: Inter-Stimulus Interval in
    seconds.
    :param trial_epochs: Number of epochs
    within a trial.
    :param n_trials: Number of trials.
    :param overlapped: Whether the epochs are
    overlapped or not.

    :return np.ndarray: It returns an array with
    the following shape: (n_trials, ), its elements
    represent the spent time until the indexed trial.
    '''
    if overlapped:
        trial_len = epoch_len + isi * (trial_epochs - 1)
    else:
        trial_len = epoch_len * trial_epochs
    spent_time = np.repeat(trial_len, n_trials)
    return spent_time.cumsum()


def __data_reshape(
        X: np.array,
        y: np.array,
        stim_seq: np.array,
        stim_tgt: np.array) -> Tuple[np.array, np.array,
                                     np.array, np.array,
                                     tuple]:
    '''
    This method will perform the necessary
    initial reshapes of the data.

    Example:
    ```
    n_runs, n_electr, n_trials, n_flashes, n_timesteps =\
        (24, 32, 20, 6, 32)
    X = np.random.randn(n_runs, n_electr, n_trials,
        n_flashes, n_timesteps)
    y = np.random.randn(n_runs, n_trials, n_flashes)
    stim_seq = np.random.randn(n_runs, n_trials, n_flashes)
    stim_tgt = np.random.randn(n_runs)

    X, y, stim_seq, stim_tgt, old_shape = __data_reshape(
        X, y, stim_seq, stim_tgt)
    X.shape, y.shape, stim_seq.shape, stim_tgt.shape, old_shape
    ----- Output -----
    ((24, 120, 1024), (24, 120), (24, 120), (24,), (24, 32, 20, 6, 32))
    ```

    :param X: Data with shape:
        (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
    :param y: Targets with shape:
        (n_runs, n_trials, n_flashes)
    :param stim_seq: The stimulus sequence
    with shape:
        (n_runs, n_trials, n_flashes)
    :param stim_tgt: And the stimulus targets
    with shape:
        (n_runs,)
    :return Tuple[np.array x4, tuple]: A tuple with (respectively)
    the run's data with shape:
        (n_runs, n_epochs, n_features)
    The targets with shape:
        (n_runs, n_epochs)
    The stimulus sequence with shape:
        (n_runs, n_epochs)
    And the stimulus targets with shape:
        (n_runs,)
    And the old shape of `X`:
        (n_runs, n_electr, n_trials, n_flashes, n_timesteps)
    '''
    # Neural Networks pre-processing
    # Old shape
    old_shape = X.shape
    n_runs, n_electr, n_trials, \
        n_flashes, n_timesteps = old_shape

    n_epochs = n_trials * n_flashes
    n_features = n_electr * n_timesteps

    # Reshape - Merge n_trials * n_flashes = n_epochs
    X = X.reshape(
        n_runs, n_electr,
        n_epochs, n_timesteps
    )
    y = y.reshape(
        n_runs, n_epochs)
    stim_seq = stim_seq.reshape(
        n_runs, n_epochs)

    # Reshape - Merge n_electr * n_timesteps = n_features
    X = X.swapaxes(1, 2)  # To avoid mixing dimensions up
    X = X.reshape(
        n_runs, n_epochs,
        n_features
    )

    return X, y, stim_seq, stim_tgt, old_shape
