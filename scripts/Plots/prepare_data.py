'''
This script will prepare the data from a series of
pre-defined methods' directories for the final
results visualisation.

By preparing we mean that our final results dictionary
will have the following structure:
```
{
    'method_name': {
        'bar_GAIN': [S1_bar_GAIN, ..., S9_bar_GAIN],
        'std_GAIN': [S1_std_GAIN, ..., S9_std_GAIN],
        'bar_CONS': [S1_bar_CONS, ..., S9_bar_CONS],
        'std_CONS': [S1_std_CONS, ..., S9_std_CONS],
        'bar_bar_t': [S1_bar_bar_t, ..., S9_bar_bar_t],
        'param_name': 'param_name',
        'bar_bar_param': [S1_bar_bar_param, ..., S9_bar_bar_param]
    },
    ...
}
```

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 26/04/2024
'''

import os
import argparse
import pickle as pkl
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from constants import HOFF_SUBJECTS

METHODS_DIRS = [
    "First Coincidence",
    "Fixed Parameters",
    "Gain - Loss balance",
    "Baseline"
]


def find_files(
        filename: str,
        search_path: str) -> Dict[str, List[str]]:
    '''
    This method will search for a file in every
    folder (and the folders contained in it).

    :param filename: The file which we are searching.
    :param search_path: Root directory in which we will
    search.

    :return Dict[str, str]: A dictionary whose keys are
    the subjects and the values a List with the methods
    results' paths.
    '''
    results_path = defaultdict(list)
    for root, _, files in os.walk(
            search_path, topdown=False):
        if filename in files:
            subject = root.split('/')[-1]
            results_path[subject].append(
                os.path.join(root, filename))
    return results_path


parser = argparse.ArgumentParser()

parser.add_argument(
    "-id", "--input-dir-path", type=str,
    help="Path to the root directory containing all the methods' results.",
    required=True, dest="in_dir_path"
)

parser.add_argument(
    "-o", "--output-path", type=str,
    help="Path to the file in which we will save all.",
    required=True, dest="out_path"
)

if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    in_dir = Path(args.in_dir_path)

    # Useful vars
    subjects = [f"S{i}" for i in HOFF_SUBJECTS]

    # Search for the results
    results_paths = defaultdict(list)
    for method_dir in METHODS_DIRS:
        more_results_paths = find_files(
            "results.pkl", in_dir / method_dir)

        results_paths = {
            sbj: results_paths[sbj] +
            more_results_paths[sbj]
            for sbj in subjects
        }

    # Load the results
    results = defaultdict(
        lambda: defaultdict(list))
    for sbj in subjects:
        for path in results_paths[sbj]:
            with open(path, 'rb') as f:
                report = pkl.load(f)

                # Parameter name
                results[report["method_name"]][
                    "param_name"] = report["param_name"]

                # Metrics
                results[report["method_name"]][
                    "bar_GAIN"].append(report["bar_GAIN"])
                results[report["method_name"]][
                    "std_GAIN"].append(report["std_GAIN"])

                results[report["method_name"]][
                    "bar_LOSS"].append(report["bar_LOSS"])
                results[report["method_name"]][
                    "std_LOSS"].append(report["std_LOSS"])

                results[report["method_name"]][
                    "bar_CONS"].append(report["bar_CONS"])
                results[report["method_name"]][
                    "std_CONS"].append(report["std_CONS"])

                # Parameters & extra info
                results[report["method_name"]][
                    "bar_bar_t"].append(report["bar_bar_t"])
                results[report["method_name"]][
                    "bar_bar_param"].append(report["bar_bar_param"])

    # Save the results
    results = dict(results)  # To make the object serializable
    os.makedirs(os.path.dirname(
        args.out_path), exist_ok=True)
    with open(args.out_path, "wb") as f:
        pkl.dump(results, f)
