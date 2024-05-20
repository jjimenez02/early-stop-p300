'''
This script will compare the average metric
obtained per method in one plot.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 28/04/2024
'''

import re
import math
import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from plots import save_plot
import matplotlib.pyplot as plt
from constants import (HOFF_SUBJECTS,
                       METHOD_NAME_BASELINE,
                       LINESTYLES,
                       COLORS,
                       METRICS_FULL_NAMES)
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

    # Obtain metrics per method
    plt.figure(figsize=(10, 10))
    all_methods = np.asarray(best_methods + [
        METHOD_NAME_BASELINE
    ]) if args.bm else np.asarray(best_methods)
    for method_idx, method in enumerate(all_methods):
        if args.metric == "GAIN+CONS÷2":
            bar_gain = np.array(
                report[method]["bar_GAIN"])
            bar_cons = np.array(
                report[method]["bar_CONS"])

            bar_metric = (bar_gain + bar_cons)/2
            std_metric = np.repeat(0, len(HOFF_SUBJECTS))
        else:
            bar_metric = np.array(
                report[method][f"bar_{args.metric}"])
            std_metric = np.array(
                report[method][f"std_{args.metric}"])

        plt.plot(
            HOFF_SUBJECTS,
            bar_metric,
            label=get_acronym(method),
            color=COLORS[method_idx],
            linestyle=LINESTYLES[method_idx]
        )

        plt.plot(
            HOFF_SUBJECTS,
            bar_metric + std_metric,
            color=COLORS[method_idx],
            linestyle=LINESTYLES[method_idx],
            alpha=.15
        )

        plt.fill_between(
            HOFF_SUBJECTS,
            bar_metric + std_metric,
            bar_metric - std_metric,
            color=COLORS[method_idx],
            alpha=.05,
            linestyle=LINESTYLES[method_idx]
        )

        plt.plot(
            HOFF_SUBJECTS,
            bar_metric - std_metric,
            color=COLORS[method_idx],
            linestyle=LINESTYLES[method_idx],
            alpha=.15
        )

    # Aesthetic
    plt.xticks(
        HOFF_SUBJECTS,
        labels=HOFF_SUBJECTS,
        rotation=45
    )
    plt.xlabel("Subjects")
    plt.ylabel(
        METRICS_FULL_NAMES[args.metric])
    plt.title(f"{METRICS_FULL_NAMES[args.metric]} scores")

    plt.grid()
    plt.legend(
        loc="best",
        ncol=math.ceil(
            len(all_methods)/3),
        fontsize="small"
    )

    plt.ylim(0, 1)
    plt.tight_layout()

    # Save plots
    save_plot(str(
        out_dir_path /
        f"{METRICS_FULL_NAMES[args.metric]}"
    ) + ".png")
