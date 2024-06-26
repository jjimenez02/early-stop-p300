'''
This module will define some constant values.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 02/02/2024
'''
import numpy as np
from scipy.stats import (wilcoxon,
                         mannwhitneyu,
                         ttest_ind,
                         ks_2samp)

SEED = 1234

HOFF_SFREQ = 2048

HOFF_NORMAL_TRIAL_EPOCHS = 5
HOFF_ANOMAL_TRIAL_EPOCHS = 1
HOFF_TRIAL_EPOCHS = HOFF_NORMAL_TRIAL_EPOCHS +\
    HOFF_ANOMAL_TRIAL_EPOCHS
HOFF_RUN_TRIALS = 20
# @hoffmann_efficient_2008 inter-stimulus interval in ms
HOFF_ISI = 400

# Subject 5 is not included
HOFF_SUBJECTS = list(range(1, 5)) +\
    list(range(6, 9 + 1))
HOFF_SESSIONS = list(range(1, 4 + 1))
HOFF_RUNS = list(range(1, 6 + 1))
HOFF_DAY_SESSIONS = 2
HOFF_DAYS = [1, 2]

FIGS_DIR = "figs/"

# More utilities
FONT_SIZE = 30

LINESTYLES = [
    (0, (1, 10)),
    (0, (1, 1)),
    (5, (10, 3)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1))
]

COLORS = [
    "#000000",
    "#000080",
    "#43FF00",
    "#808080",
    "#800000",
    "#FF0000",
    "#800080",
    "#FF00FF",
    "#008000",
    "#00FF00",
    "#808000",
    "#FFFF00",
]


# Statistical tests
def ttest_welch(a, b, **kwargs):
    return ttest_ind(a, b, equal_var=False, **kwargs)


def custom_wilcoxon(a, b, **kwargs):
    '''
    This wrapper is done in order to avoid
    errors when all the differences between both
    populations are zeroes, but the underlying
    functionality is the same as expected by
    Wilcoxon.

    In particular, we avoid doing the Wilcoxon
    stat test of two equal populations, instead
    we directly write a p-value of "1" for that
    scenario.
    '''
    n_runs, _, n_flashes = b.shape

    statistic = np.zeros(
        (n_runs, n_flashes))
    pvalue = np.zeros(
        (n_runs, n_flashes))

    # See what stimulus are repeated
    matches = np.all(a == b, axis=(0, 1))

    # We assign the p-vals for the repeated column
    pvalue[:, matches] = 1

    # We remove "a" stimuli from "b"
    b_exclusive = b[:, :, ~matches]

    # Perform the test
    stats = wilcoxon(
        a, b_exclusive, **kwargs)

    # Save the values
    statistic[:, ~matches] += stats.statistic
    pvalue[:, ~matches] += stats.pvalue

    # We simulate a WilcoxonResult object
    wilcoxon_result = type(
        'WilcoxonResult', (), {})()
    wilcoxon_result.statistic = statistic
    wilcoxon_result.pvalue = pvalue

    return wilcoxon_result


AVAILABLE_TESTS = [
    custom_wilcoxon, ttest_ind,
    mannwhitneyu, ttest_welch,
    ks_2samp
]

# Metrics
METRICS_FULL_NAMES = {
    "GAIN": "Ganancia",
    "LOSS": "Pérdida",
    "CONS": "Conservación",
    "GAIN+CONS÷2": "(Ganancia + Conservación) ÷ 2"
}

METRIC_COLORS = {
    "GAIN": "#0099ff",
    "LOSS": "#ff9933",
    "CONS": "#33cc33",
    "GAIN+CONS÷2": "#00ffcc"
}

# Methods names
METHOD_NAME_BASELINE = "Baseline method"

# P300 colors
COLOR_P300 = COLORS[0]
COLOR_NON_P300 = COLORS[-1]

# Labels
IS_TGT = 1

# Control & Disabled names
SBJ_NAME_CTRL = "Sin problemas conocidos"
SBJ_NAME_DSBL = "Con problemas neurodegenerativos"
