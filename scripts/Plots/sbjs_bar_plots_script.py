'''
This script will compare the average metric
obtained per method in one bar plot per subject.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 28/04/2024
'''

import re
import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from constants import (HOFF_SUBJECTS,
                       METHOD_NAME_BASELINE,
                       METRICS_FULL_NAMES,
                       METRIC_COLORS)
from plots import (scores_bar_plot,
                   save_plot)
from report_utils import (get_acronym,
                          obtain_n_best_methods)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input-path", type=str,
    help="Path to the file containing the data",
    required=True, dest="in_path"
)

parser.add_argument(
    "-od", "--output-dir-path", type=str,
    help="Path to the directory containing the plots",
    required=True, dest="out_dir_path"
)

parser.add_argument(
    "-m", "--metric", type=str,
    help="Metric to plot",
    choices=METRICS_FULL_NAMES.keys(),
    required=True, dest="metric"
)

parser.add_argument(
    "-n", "--number-of-methods", type=int,
    help="Number of best methods to consider, if it is -1 it plot all.",
    default=-1,
    required=False, dest="n"
)

parser.add_argument(
    "-reg", "--regexp", type=str,
    help="Regexp expression to filter the methods.",
    default=".",
    required=False, dest="regexp"
)

parser.add_argument(
    "-bm", "--baseline-method",
    help="This option will also include the baseline method.",
    action="store_true",
    dest="bm",
    default=False
)

if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    out_dir_path = Path(args.out_dir_path)

    # Load report
    with open(args.in_path, "rb") as f:
        report = pkl.load(f)

    # Filter methods
    methods_names = report.keys()
    filtered_methods = [
        i for i in methods_names
        if re.match(args.regexp, i)
    ]

    if len(filtered_methods) == 0:
        raise Exception("No methods found")

    # Consider just the best methods
    best_methods = obtain_n_best_methods(
        filtered_methods, report,
        criteria=lambda method: np.sum(
            method["bar_GAIN"] +
            method["bar_CONS"])/2,
        n=args.n
    )

    # Sorted barplots (by score) by subject
    all_methods = np.asarray(best_methods + [
        METHOD_NAME_BASELINE
    ]) if args.bm else np.asarray(best_methods)
    for sbj_idx, sbj in enumerate(HOFF_SUBJECTS):
        # Obtain metrics per method
        bar_metrics, std_metrics =\
            np.zeros(len(all_methods)), np.zeros(len(all_methods))
        for method_idx, method in enumerate(all_methods):
            if args.metric == "GAIN+CONS÷2":
                bar_gain = np.array(
                    report[method]["bar_GAIN"])[sbj_idx]
                bar_cons = np.array(
                    report[method]["bar_CONS"])[sbj_idx]

                bar_metrics[method_idx] = (bar_gain + bar_cons)/2
                std_metric = np.repeat(0, len(HOFF_SUBJECTS))
            else:
                bar_metrics[method_idx] =\
                    report[method][f"bar_{args.metric}"][sbj_idx]
                std_metrics[method_idx] =\
                    report[method][f"std_{args.metric}"][sbj_idx]

        # Sort metrics
        metric_sorted_idxs = np.flip(
            np.argsort(bar_metrics))

        # Sorted bar plots
        scores_bar_plot(
            names=[
                get_acronym(method) for method in
                all_methods[metric_sorted_idxs]
            ],
            scores={
                METRICS_FULL_NAMES[args.metric]:
                    bar_metrics[metric_sorted_idxs],
            },
            colors={
                METRICS_FULL_NAMES[args.metric]:
                    METRIC_COLORS[args.metric]
            },
            xlabel="Methods",
            ylabel=METRICS_FULL_NAMES[args.metric],
            title=f"{METRICS_FULL_NAMES[args.metric]} scores for S{sbj}",
            ymin=0,
            yerr={
                METRICS_FULL_NAMES[args.metric]:
                    std_metrics[metric_sorted_idxs],
            },
            rotation=45
        )
        save_plot(str(
            out_dir_path /
            f"{METRICS_FULL_NAMES[args.metric]} - S{sbj}"
        ) + ".png")
