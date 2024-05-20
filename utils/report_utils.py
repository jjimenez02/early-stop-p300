'''
This module will contain some utilities
to save results in dictionaries, parse
them, ...

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 24/04/2024
'''

import numpy as np
from typing import List, Dict
from voting_utils import avg_diffs


def es_extra_vals(
        bar_val_list: List[float],
        std_val_list: List[float],
        bar_ref_list: List[float],
        std_ref_list: List[float],
        y_pred: np.ndarray,
        z_pred: np.ndarray,
        z_test: np.ndarray,
        t_pred: np.ndarray,
        t_ref: int) -> None:
    '''
    This method will obtain a series of values
    as a result of a series of statistics to obtain
    its mean and standard deviation and append them
    at the end of `bar_val_list` and `std_val_list`,
    respectively.

    The same will be done for a reference value to
    compare them in the future.

    :param bar_val_list: List in which we will append
    the mean of the statistic extracted.
    :param std_val_list: List in which we will append
    the standard deviation of the statistic extracted.
    :param bar_ref_list: List in which we will append
    the mean of the reference statistic.
    :param std_ref_list: List in which we will append
    the standard deviation of the reference statistic.
    :param y_pred: 3D numpy array with predictions and
    shape: (n_runs, n_trials, n_flashes)
    :param z_pred: 2D numpy array with predicted stimulus
    indexes and shape: (n_runs, n_trials)
    :param z_test: 2D numpy array with real stimulus
    indexes and shape: (n_runs, n_trials)
    :param t_pred: 1D numpy array with the index per
    run at which an early stop has been predicted,
    shape: (n_runs,)
    :param t_ref: Reference trial index at which we fix-stop.
    '''
    n_runs, _, _ = y_pred.shape
    run_idxs = np.arange(n_runs)

    # Extract predictions
    z_t_pred = z_pred[run_idxs, t_pred]
    z_t_ref = z_test[:, t_ref]

    # Extract averaged differences
    y_diff = avg_diffs(y_pred)

    # Chosen statistics
    chosen_vals = y_diff[
        run_idxs, t_pred, z_t_pred]

    bar_val = chosen_vals.mean()
    std_val = chosen_vals.std()

    bar_val_list.append(bar_val)
    std_val_list.append(std_val)

    # Referent statistics
    ref_vals = y_diff[
        run_idxs, t_ref, z_t_ref]

    bar_ref = ref_vals.mean()
    std_ref = ref_vals.std()

    bar_ref_list.append(bar_ref)
    std_ref_list.append(std_ref)


def es_save_vals(
        results: dict,
        method_name: str,
        param_name: str,
        bar_bar_t: float,
        bar_bar_param: float,
        gain_vals: List[float],
        loss_vals: List[float]) -> None:
    '''
    This method will save some values within the
    `results` dictionary.

    :param results: A dictionary in which we will save
    a series of statistics.
    :param method_name: Method's name.
    :param param_name: Parameter's name.
    :param bar_bar_t: Averaged trial at which we stopped.
    :param bar_bar_param: Averaged parameter optimised.
    :param gain_vals: List of gains.
    :param loss_vals: List of losses.
    '''
    results["bar_GAIN"] = np.mean(gain_vals)
    results["std_GAIN"] = np.std(gain_vals)

    results["bar_LOSS"] = np.mean(loss_vals)
    results["std_LOSS"] = np.std(loss_vals)

    # Conservation
    cons_vals = 1 - np.asarray(loss_vals)
    results["bar_CONS"] = np.mean(cons_vals)
    results["std_CONS"] = np.std(cons_vals)

    results["bar_bar_t"] = bar_bar_t
    results["bar_bar_param"] = bar_bar_param

    results["method_name"] = method_name
    results["param_name"] = param_name


def es_save_extra_vals(
        results: dict,
        bar_val_list: List[float],
        bar_ref_list: List[float]) -> None:
    '''
    This method will save the extra values within the
    `results` dictionary.

    :param results: A dictionary in which we will save
    a series of statistics.
    :param bar_val_list: List from which we will extract
    some statistics.
    :param bar_ref_list: List from which we will extract
    some statistics.
    '''
    results["bar_bar_chosen_val"] =\
        np.mean(bar_val_list)
    results["std_bar_chosen_val"] =\
        np.std(bar_val_list)

    results["bar_bar_ref_val"] =\
        np.mean(bar_ref_list)
    results["std_bar_ref_val"] =\
        np.std(bar_ref_list)


def es_save_session_vals(
        results: dict,
        session_idx: int,
        bar_t: float,
        bar_param: float,
        gain_val: float,
        loss_val: float,
        score_ref: float,
        score_cur: float) -> None:
    '''
    This method will save some values within the
    `results` dictionary.

    :param results: A dictionary in which we will save
    a series of statistics.
    :param session_idx: Current session's index.
    :param bar_t: Averaged trial at which we stopped.
    :param bar_param: Averaged parameter optimised.
    :param gain_val: Gain value.
    :param loss_val: Loss value.
    :param score_ref: Reference's score (used to obtain the loss).
    :param score_cur: Algorithm's score.
    '''
    results[f"Session_{session_idx}_Out"] = {
        "bar_t": bar_t,
        "bar_param": bar_param,
        "GAIN": gain_val,
        "LOSS": loss_val,
        "CONS": 1 - loss_val,
        "score_ref": score_ref,
        "score_cur": score_cur
    }


def es_save_session_extra_vals(
        results: dict,
        session_idx: int,
        bar_val_list: List[float],
        std_val_list: List[float],
        bar_ref_list: List[float],
        std_ref_list: List[float]):
    '''
    This method will save the extra values within the
    `results` dictionary.

    :param results: A dictionary in which we will save
    a series of statistics.
    :param session_idx: Current session's index.
    :param bar_val_list: List from which we will extract
    the averaged results.
    :param std_val_list: List from which we will extract
    the standard deviation results.
    :param bar_ref_list: List from which we will extract
    the averaged results for the referent.
    :param std_ref_list: List from which we will extract
    the standard deviation results for the referent.
    '''
    key = f"Session_{session_idx}_Out"

    results[key]["bar_chosen_val"] =\
        bar_val_list[-1]
    results[key]["std_chosen_val"] =\
        std_val_list[-1]

    results[key]["bar_ref_val"] =\
        bar_ref_list[-1]
    results[key]["std_ref_val"] =\
        std_ref_list[-1]


def obtain_n_best_methods(
        methods: List[str],
        report: Dict[str, Dict[str, List[float]]],
        criteria: callable,
        n: int) -> List[str]:
    '''
    This method will obtain the first N-th
    methods given the `criteria` function.

    Example:
    ```
    np.random.seed(1234)

    n_vals = 2

    methods = ["A", "B", "C"]
    report = {
        "A": {
            "bar_X": np.ones(n_vals)*2,
            "bar_Y": np.ones(n_vals)*2
        },
        "B": {
            "bar_X": np.zeros(n_vals),
            "bar_Y": np.zeros(n_vals)
        },
        "C": {
            "bar_X": np.ones(n_vals),
            "bar_Y": np.ones(n_vals)
        }
    }

    best_methods = obtain_n_best_methods(
        methods, report,
        criteria=lambda method: np.sum(
            method["bar_X"] + method["bar_Y"])/2,
        n=2
    )

    print("BEST METHODS:")
    print(best_methods)
    ----- Output -----
    BEST METHODS:
    ['A', 'C']
    ```

    :param methods: List of methods' names.
    :param report: Dictionary with all the methods
    and its values.
    :param criteria: Criteria to score every method,
    the function should have the following signature:
    `def fun(method: Dict[str, List[float]])`
    Note: the higher the score (& positive), the better.
    :param n: N methods to choose, if n<0 it will
    return all the methods, but sorted.
    :return List[str]: List with the N-th best methods'
    names given the `criteria` function.
    '''
    scores = np.array([
        criteria(report[method])
        for method in methods
    ])

    sorted_idxs = np.flip(
        np.argsort(scores))
    sorted_methods = [
        methods[idx] for idx in sorted_idxs]

    if n < 0:
        return sorted_methods
    else:
        return sorted_methods[:n]


def get_acronym(input_string: str):
    '''
    This method will build an acronym
    from the given string. Specifically
    it will upper case the whole string
    and get the first letter of every word.

    Example:
    ```
    get_acronym("hello - world 2024asd dsa")
    ----- Output -----
    'H-W2D'
    ```

    :param input_string: String to
    transform into an acronym.
    '''
    # Split the string into words
    words = list()
    bow = list()
    bow_tmp = list()
    for word in input_string.split():
        bow.append(word)

        # We consider dyphens
        for subword in bow:
            if '-' in subword:
                dyphen_subwords = subword.split('-')
                if not dyphen_subwords == ['', '']:
                    bow_tmp += dyphen_subwords
                else:
                    bow_tmp += ['-']
            else:
                bow_tmp.append(subword)
        bow.clear()

        bow += bow_tmp
        bow_tmp.clear()

        for subword in bow:
            if '_' in subword:
                unders_words = subword.split('_')
                bow_tmp += unders_words
            else:
                bow_tmp.append(subword)
        bow.clear()

        bow += bow_tmp
        bow_tmp.clear()

        words += bow
        bow.clear()

    # Take the first letter of each word and make them uppercase
    first_letters = [word[0].upper() for word in words if word]
    # Join the first letters to form the acronym
    acronym = ''.join(first_letters)
    return acronym
