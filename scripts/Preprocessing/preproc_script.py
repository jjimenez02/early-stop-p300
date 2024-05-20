'''
This script will apply a regular pre-processing
over @hoffmann_efficient_2008 data, specifically:
· Mastoids averaging
· FW-BW Butterworth Filter 1-12Hz
· Downsampling
· Epochs extraction
· Electrodes selection
· Pick up just the first 20 trials to ease data
access.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 06/02/2024
'''

import os
import math
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
from typing import List, Dict
from HoffElect import HoffElect
from scipy.signal import (butter,
                          sosfiltfilt)
from path_utils import get_run_path
from constants import (HOFF_SUBJECTS,
                       HOFF_SESSIONS,
                       HOFF_RUNS,
                       HOFF_SFREQ,
                       HOFF_ISI,
                       HOFF_TRIAL_EPOCHS,
                       HOFF_RUN_TRIALS)

vinicio_best_electrodes = {
    1: [HoffElect.FP1.value, HoffElect.CP5.value,
        HoffElect.P3.value, HoffElect.OZ.value,
        HoffElect.PO4.value],
    2: [HoffElect.CP1.value, HoffElect.P7.value,
        HoffElect.P3.value, HoffElect.PO3.value],
    3: [HoffElect.CP1.value, HoffElect.P7.value,
        HoffElect.P3.value, HoffElect.PO3.value],
    4: [HoffElect.CP5.value, HoffElect.P7.value,
        HoffElect.O2.value, HoffElect.CP6.value,
        HoffElect.FC2.value],
    6: [HoffElect.P7.value, HoffElect.PO3.value,
        HoffElect.O1.value, HoffElect.O2.value,
        HoffElect.P8.value],
    7: [HoffElect.CP5.value, HoffElect.P3.value,
        HoffElect.PZ.value, HoffElect.PO3.value,
        HoffElect.PO4.value, HoffElect.P8.value],
    8: [HoffElect.CP5.value, HoffElect.P7.value,
        HoffElect.P3.value, HoffElect.PO4.value,
        HoffElect.CP6.value],
    9: [HoffElect.P3.value, HoffElect.PZ.value,
        HoffElect.OZ.value, HoffElect.O2.value]
}

hoffmann_8_set = {
    subject_nr: [
        HoffElect.FZ.value,
        HoffElect.CZ.value,
        HoffElect.PZ.value,
        HoffElect.P3.value,
        HoffElect.P7.value,
        HoffElect.OZ.value,
        HoffElect.P4.value,
        HoffElect.P8.value
    ] for subject_nr in HOFF_SUBJECTS
}

all_elects = list(
    HoffElect.__members__.values())
# Remove mastoids
all_elects.remove(HoffElect.M1)
all_elects.remove(HoffElect.M2)
# Get indexes
all_elects = list(map(
    lambda x: x.value, all_elects))

all_electrodes = {
    subject_nr: all_elects
    for subject_nr in HOFF_SUBJECTS
}


def bandpass_filter(X: np.ndarray) -> np.ndarray:
    """
    Forward-Backward third order Butterworth
    bandpass filter between 1 & 12 Hz.

    :param X:The input signal matrix,
    expected shape: (n_feats, n_timesteps)
    :return np.array: The filtered
    signal matrix.
    """
    # Define the filter parameters
    order = 3
    Wn = [1/1024, 12/1024]  # cutoff frequencies
    btype = 'bandpass'      # filter type (forward-backward Butterworth)
    output = 'sos'           # output format (second-order sections)

    # Create the filter object
    sos = butter(
        order, Wn,
        btype=btype,
        output=output
    )

    # Apply the filter to the signals
    X_filtered = sosfiltfilt(
        sos, X, axis=1)

    return X_filtered


def hoffmann_events_to_date_array(
        x: np.array) -> np.array:
    '''
    This method will receive a numpy array
    with a specific format:
        [YYYY, MM, DD, HH, mm, ss.SSS]

    And will be transformed into a numpy's
    datetime array to perform date substractions.

    We assume the data used as input follows
    the format of @hoffman_efficient_2008
    (10.1016/j.jneumeth.2007.03.005).

    :param x: Numpy array with the dates.
    :return np.array: A numpy's datetime array.
    '''
    # Initialize an empty array to hold the datetime64 objects
    dates = np.empty(
        x.shape[0], dtype='datetime64[ms]')

    # Convert each row in the array to a datetime64 object
    for i in range(x.shape[0]):
        dates[i] = np.datetime64(
            f"{int(x[i, 0])}-{int(x[i, 1]):02d}-" +
            f"{int(x[i, 2]):02d}T{int(x[i, 3]):02d}:" +
            f"{int(x[i, 4]):02d}:{x[i, 5]:06.3f}"
        )

    return dates


def preproc_data(
        data: dict,
        subject_nr: int,
        elect_set: Dict[int, List[HoffElect]],
        desired_sfreq: int,
        window_len: float = 1) -> np.array:
    '''
    This method will preprocess all the data
    following a similar approach to @hoffmann_efficient_2008.

    :param data: Dictionary with
    all the data.
    :param subject_nr: Subject identifier.
    :param elect_set: Chosen electrodes' set.
    :param desired_sfreq: Desired sample's frequency
    after downsampling.
    :param window_len: Epoch's length in seconds,
    defaults to one.
    :return np.array: Array with all the
    data, dimensions:
    (n_electr, 20 trials, 6 epochs, timesteps)
    '''
    run = data["data"]

    # Referencing & dropping mastoids
    X_run = run[:32] - \
        np.mean(run[[HoffElect.M1.value, HoffElect.M2.value]], axis=0)

    # Band-pass filtering
    X_run = bandpass_filter(X_run)

    # Down-sampling
    X_run = X_run[:, ::HOFF_SFREQ//desired_sfreq]

    # Stimulus onset indexes
    events = hoffmann_events_to_date_array(data["events"])
    e_secs = (events - events[0]).astype(float)/1e3  # ms2s
    offsets = np.round((
        e_secs * desired_sfreq +
        # ms2sec
        HOFF_ISI/1e3 * desired_sfreq)).astype(int)

    # Epochs' extraction
    window_size = math.ceil(window_len * desired_sfreq)  # seconds x Hz
    windows = [X_run[:, i:i+window_size] for i in offsets]

    # Swap to have (n_electr, n_epochs, n_timesteps)
    X_run = np.asarray(windows).swapaxes(0, 1)

    # Choose electrodes
    X_run = X_run[elect_set[subject_nr]]

    # Reshape into (n_electr, n_trials, 6, n_timesteps)
    X_run = X_run.reshape((
        X_run.shape[0], X_run.shape[1]//HOFF_TRIAL_EPOCHS,
        HOFF_TRIAL_EPOCHS, X_run.shape[2]
    ))

    # Pick up just the first 20 trials
    n_trials = HOFF_RUN_TRIALS
    X_run = X_run[:, :n_trials]

    return X_run


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
    help="Path to the directory containing the epoched",
    required=True, dest="out_dir_path"
)

parser.add_argument(
    "-es", "--electrodes-set", type=str,
    help="Electrodes' set to use",
    required=False, dest="elect_set",
    choices=[
        "vinicio_best_electrodes",
        "hoffmann_8_set",
        "all_electrodes"
    ], default="all_electrodes"
)

parser.add_argument(
    "-el", "--epoch-length", type=float,
    help="Epoch's length in seconds",
    required=False, dest="epoch_len",
    default=1
)

parser.add_argument(
    "-sfreq", "--sampling-frequency", type=int,
    help="Desired sampling frequency after downsampling",
    required=False, dest="sfreq",
    default=32
)

if __name__ == "__main__":
    args = parser.parse_args()

    electrodes_sets = {
        "vinicio_best_electrodes": vinicio_best_electrodes,
        "hoffmann_8_set": hoffmann_8_set,
        "all_electrodes": all_electrodes
    }

    for session_nr in tqdm(HOFF_SESSIONS):
        for run_nr in HOFF_RUNS:
            # Load run data
            data = loadmat(get_run_path(
                args.in_dir_path, args.subject_nr,
                session_nr, run_nr),
                simplify_cells=True
            )

            # Data's preprocessing
            X_run = preproc_data(
                data, args.subject_nr,
                electrodes_sets[args.elect_set],
                args.sfreq, args.epoch_len
            )

            # Stimulus Sequences' extraction
            stim_seq = data["stimuli"] - 1  # Matlab Index
            stim_tgt = data["target"] - 1  # Matlab Index

            # Reshape from (n_epochs,) into (n_trials, 6)
            stim_seq = stim_seq.reshape(
                len(stim_seq)//HOFF_TRIAL_EPOCHS,
                HOFF_TRIAL_EPOCHS
            )

            # Pick up just the first 20 trials
            stim_seq = stim_seq[:HOFF_RUN_TRIALS]

            # Target's extraction
            y_run = (stim_seq == stim_tgt).astype(int)

            # Save run's epoched data
            out_path = get_run_path(
                args.out_dir_path, args.subject_nr,
                session_nr, run_nr, empty_dir=True
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                pkl.dump((X_run, y_run, stim_seq, stim_tgt), f)

    # Add metadata
    subject_dir_path = Path(
        args.out_dir_path) / f"subject{args.subject_nr}"

    elect_idxs = electrodes_sets[args.elect_set][args.subject_nr]
    elect_names = np.array(list(HoffElect.__members__))
    with open(subject_dir_path / "electrodes.pkl", "wb") as f:
        pkl.dump(elect_names[elect_idxs], f)
