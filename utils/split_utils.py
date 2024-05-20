'''
This module will define some utilites
when splitting the data.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 15/02/2024
'''

import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import LeaveOneGroupOut


def get_logo_idxs(
        run_ids: np.array) -> Tuple[np.array, np.array]:
    '''
    This method takes in an array of run IDs and
    returns two arrays of indices for training and
    testing data.

    The function uses the LeaveOneGroupOut
    cross-validation strategy, which leaves
    one group (i.e., one set of runs) out of the
    dataset randomly.

    Example:
    ```
    np.random.seed(1234)

    run_ids = np.repeat(
        np.arange(len(HOFF_SESSIONS)),
        len(HOFF_RUNS)
    )
    train_idxs, test_idxs = get_logo_idxs(run_ids)
    train_idxs, test_idxs
    ----- Output -----
    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17]),
     array([18, 19, 20, 21, 22, 23]))
    ```

    :param run_ids: 1D-Array with the identifiers.
    :return Tuple[np.array x2]: The function returns
    the train_idxs and test_idxs arrays.
    '''
    # Leave-One-Session-Out
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=run_ids)
    split = np.random.randint(0, n_splits)

    for i, (train, test) in enumerate(
            logo.split(run_ids, groups=run_ids)):
        train_idxs, test_idxs = train, test
        if i == split:
            break

    return train_idxs, test_idxs


def get_sbj_runs_ids(
        n_sessions: int,
        n_runs: int) -> np.ndarray:
    '''
    This method will return a list with each
    run's identifiers.

    Warning: This method will assume that runs from
    the same session are contiguous.

    Example:
    ```
    get_sbj_runs_ids(4, 6)
    ----- Output -----
    array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
       3, 3])
    ```

    :param n_sessions: Number of sessions per subject.
    :param n_runs: Number of runs per session.
    :return np.ndarray: Array with the runs' identifiers.
    '''
    return np.repeat(
        np.arange(n_sessions), n_runs)


def get_sbj_runs_daily_ids(
        n_sessions: int,
        n_runs: int,
        n_days: int) -> Tuple[
            np.ndarray, Dict[int, np.ndarray]]:
    '''
    This method will return the identifiers for
    every run and a dictionary with the correspondences
    between day and session.

    Warning: This method will assume that runs from
    the same session are contiguous.

    Example:
    ```
    get_sbj_runs_daily_ids(4, 6, 2)
    ----- Output -----
    (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
            3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7,
            7, 7, 7, 7]),
     {0: array([0, 1, 2, 3]), 1: array([4, 5, 6, 7])})
    ```

    :param n_sessions: Number of sessions per day.
    :param n_runs: Number of runs per session.
    :param n_days: Number of days measured for a subject.
    :return Tuple[np.ndarray, Dict[int, np.ndarray]]: Tuple
    with an array of run identifiers and a dictionary with the
    day number as key and a list of session identifiers as value.
    '''
    days_dict = {}

    runs_ids = get_sbj_runs_ids(
        n_sessions*n_days, n_runs)
    uniq_ids = np.unique(runs_ids)

    rel_pos = 0
    for day_nr in range(n_days):
        days_dict[day_nr] = uniq_ids[
            rel_pos:rel_pos + n_sessions]
        rel_pos += n_sessions

    return runs_ids, days_dict


def get_daily_idxs(
        runs_ids: np.ndarray,
        days_dict: Dict[int, np.ndarray],
        test_size: float,
        shuffle: bool = False) -> Tuple[
            np.ndarray, np.ndarray]:
    '''
    This method will split the runs set into
    training & testing sub-sets in function of
    the days dictionary and test size specified.

    Example:
    ```
    _ids, _dict = get_sbj_runs_daily_ids(4, 6, 3)
    train_idxs, test_idxs = get_daily_idxs(_ids, _dict, .3)

    _dict, train_idxs, test_idxs
    ----- Output -----
    ({0: array([0, 1, 2, 3]),
      1: array([4, 5, 6, 7]),
      2: array([ 8,  9, 10, 11])},
     array([0, 1, 2, 3, 4, 5, 6, 7]),
     array([ 8,  9, 10, 11]))
    ```

    :param run_ids: 1D-Array with the identifiers.
    :param days_dict: A dictionary with the correspondencies
    between day and sessions identifiers (see the function
    `get_sbj_runs_daily_ids`).
    :param test_size: Number between zero and one which will
    determine how many days will be sent to the test set.
    :param shuffle: Whether to shuffle or not the days, if False
    is specified it will pick up the first for training and the last
    for testing (in function of `test_size`), defaults to `False`.
    :return Tuple[np.array x2]: The function returns
    the train_idxs and test_idxs arrays.
    '''
    days = np.array(
        list(days_dict.keys()))
    if shuffle:
        np.random.shuffle(days)

    n_train_days = int(
        len(days) * (1 - test_size))

    train_days = days[:n_train_days]
    test_days = days[n_train_days:]

    *train_sess, = map(
        lambda x: days_dict[x], train_days)
    *test_sess, = map(
        lambda x: days_dict[x], test_days)

    return np.where(np.isin(runs_ids, train_sess))[0], \
        np.where(np.isin(runs_ids, test_sess))[0]


def index_data(
        idxs: np.ndarray,
        *args: List[np.ndarray]) -> List[np.ndarray]:
    '''
    This method will index a list of arrays
    with the specified identifiers.

    :param idxs: 1D-Array with the identifiers.
    :param args: List of arrays.
    :return List[np.ndarray]: List
    with the indexed arrays.
    '''
    return list(map(lambda x: x[idxs], args))
