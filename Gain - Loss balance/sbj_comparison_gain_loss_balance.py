'''
This script will compare the average gain
and loss obtained per subject for a specific
method.

This is the variant for the Gain-Loss balance.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 16/04/2024
'''

import argparse
import numpy as np
import pickle as pkl
from pathlib import Path
from plots import (save_plot,
                   scores_bar_plot)
import matplotlib.pyplot as plt
from constants import HOFF_SUBJECTS

parser = argparse.ArgumentParser()

parser.add_argument(
    "-id", "--input-dir-path", type=str,
    help="Path to the directory containing the data",
    required=True, dest="in_dir_path"
)

parser.add_argument(
    "-od", "--output-dir-path", type=str,
    help="Path to the directory containing the plot",
    required=False, dest="out_dir_path",
    default=None
)

parser.add_argument(
    "-t", "--title", type=str,
    help="Plots title",
    required=False, dest="title",
    default=""
)

if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()
    data_path = Path(args.in_dir_path)

    # Load results
    gain_vals = np.zeros(len(HOFF_SUBJECTS))
    loss_vals = np.zeros(len(HOFF_SUBJECTS))
    gain_stds = np.zeros(len(HOFF_SUBJECTS))
    loss_stds = np.zeros(len(HOFF_SUBJECTS))

    # Top bar texts
    top_val_text = list()
    top_ref_text = list()

    axs_vals = np.zeros(len(HOFF_SUBJECTS))
    title = ""
    for sbj_idx, sbj in enumerate(HOFF_SUBJECTS):
        with open(data_path / f"S{sbj}" / "results.pkl", "rb") as f:
            report = pkl.load(f)

            # Chosen value
            bar_bar_val = np.round(
                report['bar_bar_chosen_val'],
                decimals=2
            )
            std_bar_val = np.round(
                report['std_bar_chosen_val'],
                decimals=2
            )

            top_val_text.append(
                f"AAD: {bar_bar_val} ± {std_bar_val}")

            # Reference value
            bar_bar_ref = np.round(
                report['bar_bar_ref_val'],
                decimals=2
            )
            std_bar_ref = np.round(
                report['std_bar_ref_val'],
                decimals=2
            )

            top_ref_text.append(
                f"AAD - Ref: {bar_bar_ref} ± {std_bar_ref}")

            # Load titles and mirror axis
            title = report["method_name"]
            label = report["param_name"]
            axs_vals[sbj_idx] = np.round(
                report["bar_bar_param"],
                decimals=2
            )

            # Load metrics
            gain_vals[sbj_idx] = report["bar_GAIN"]
            gain_stds[sbj_idx] = report["std_GAIN"]
            loss_vals[sbj_idx] = report["bar_LOSS"]
            loss_stds[sbj_idx] = report["std_LOSS"]

    # Plot results
    title = args.title + title
    scores_bar_plot(
        names=HOFF_SUBJECTS,
        scores={
            "Gain": gain_vals,
            "Loss": loss_vals
        },
        colors={"Gain": '#0099ff', "Loss": '#ff9933'},
        xlabel="Subjects",
        ylabel="Metric",
        title=title,
        ymin=0,
        yerr={
            "Gain": gain_stds,
            "Loss": loss_stds
        }
    )

    # Mirror axis
    for sbj_idx, sbj in enumerate(HOFF_SUBJECTS):
        plt.hlines(
            gain_vals[sbj_idx], sbj_idx, len(HOFF_SUBJECTS),
            linestyles='--', colors='grey'
        )
    axs = plt.twinx()
    axs.set_yticks(gain_vals, axs_vals)
    axs.set_ylabel(label)

    # Text at the top of every bar
    xlocs = plt.gca().get_xticks()
    ylocs = np.max([gain_vals, loss_vals], axis=0)
    for i, (val_text, ref_text) in enumerate(zip(
            top_val_text, top_ref_text)):
        plt.text(
            xlocs[i] - .15,
            ylocs[i] + .01,
            val_text,
            rotation=90
        )

        plt.text(
            xlocs[i] - .45,
            ylocs[i] + .01,
            ref_text,
            rotation=90
        )

    # Aesthethic purposes
    plt.xlim(-1, 8)

    # Save or display figure
    if args.out_dir_path is None:
        plt.show()
    else:
        save_plot(str(Path(args.out_dir_path) / title) + ".png")
