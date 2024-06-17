'''
This module will define some utilites
for the gain-loss balance optimization.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 16/04/2024
'''

import numpy as np
from pathlib import Path
from plots import save_plot
from typing import List, Any
import matplotlib.pyplot as plt
from constants import (METRICS_FULL_NAMES,
                       METRIC_COLORS,
                       FONT_SIZE)
from math_master import (find_intersections,
                         find_shared_maximum)


def obtain_gain_loss_intersection(
        gains: np.ndarray,
        losses: np.ndarray,
        ox: np.ndarray,
        xlim: List[int],
        round: bool,
        plot: bool = False,
        out_dir: Path = Path("figs"),
        title: str = "My-Plot",
        xlabel: str = "OX",
        xtick_labels: List[Any] = None,
        invert_xaxis: bool = False) -> float:
    '''
    This method will obtain the FIRST intersection
    between the Gain & Loss arrays and plot them
    if it is specified in `plot`.

    NOTE: If there is no intersection, it will return
    the value which maximises both curves (i.e:
    `np.argmax(gains * (1-losses))`)

    Example:
    ```
    tmax = 20
    xlim = [-1, 1]
    ox = np.linspace(xlim[0], xlim[1], tmax)

    gains = .5 - ox*ox
    losses = ox*ox

    print(obtain_gain_loss_intersection(
        gains, losses, ox, xlim, round=False,
        plot=True, out_dir=Path("example")
    ))
    ----- Output -----
    0.05263157894736836
    ```

    :param gains: 1D numpy array with the gain
    values, it should have the same shape as the
    `losses` array.
    :param losses: 1D numpy array with the loss
    values, it should have the same shape as the
    `gains` array.
    :param ox: 1D OX-values which bring every gain and loss
    value, it should have the same shape as the `losses`
    and `gains` arrays.
    :param xlim: Lineal space in which we will interpolate,
    specifically this will be an array with two components:
    the lower & upper bounds, respectively.
    :param round: Whether to round or not the chosen value.
    :param plot: Whether to plot or not the Gain
    VS Loss figure and its intersections, defaults to False.
    :param out_dir: Path at which we will save the figures.
    :param title: Figure's title.
    :param xlabel: OX label
    :param xtick_labels: Labels for every OX point.
    :param invert_xaxis: Whether to invert the OX axis or
    not (it will also change the intersection value chosen
    from the first to the last), default to False.

    :return float: The value at which both curves
    intersect.
    '''
    x_int, y_int = find_intersections(
        ox, ox, gains,
        1-losses, xlim
    )

    if plot:
        plt.figure(figsize=(20, 5))

    # If we do not have any intersection
    # we will pick the value which maximises
    # the product of both curves
    if len(x_int) == 0:
        t = np.argmax(
            gains * (1-losses))
        t_x = ox[t]
    else:
        idx = -1 if invert_xaxis else 0
        t_x = np.round(x_int[idx])\
            if round else x_int[idx]

    # Create a plot to see the curves
    if plot:
        plt.plot(ox, gains, label="Gain")
        plt.plot(ox, 1-losses, label="Conservation")
        plt.scatter(x_int, y_int)
        plt.vlines(
            t_x, 0, 1.1,
            label="Chosen value",
            linestyles="--"
        )

        plt.xticks(
            ox, labels=xtick_labels,
            rotation=45
        )

        plt.ylim((0, 1.1))
        plt.xlabel(xlabel)
        plt.ylabel("Metric")

        if invert_xaxis:
            plt.gca().invert_xaxis()

        plt.grid()
        plt.legend()

        plt.title(title)
        plt.tight_layout()
        save_plot(str(out_dir / title))

    return t_x


def obtain_gain_loss_shared_max(
        gains: np.ndarray,
        losses: np.ndarray,
        ox: np.ndarray,
        xlim: List[int],
        round: bool,
        plot: bool = False,
        out_dir: Path = Path("figs"),
        title: str = "My-Plot",
        xlabel: str = "OX",
        xtick_labels: List[Any] = None,
        invert_xaxis: bool = False) -> float:
    '''
    This method will obtain the shared maximum
    between the Gain & Loss arrays and plot them
    if it is specified in `plot`.

    Example:
    ```
    tmax = 20
    xlim = [-1, 1]
    ox = np.linspace(xlim[0], xlim[1], tmax)

    gains = .5 - ox*ox
    losses = ox*ox

    print(obtain_gain_loss_shared_max(
        gains, losses, ox, xlim, round=False,
        plot=True, out_dir=Path("example")
    ))

    ----- Output -----
    0.0010010010010010895
    ```

    :param gains: 1D numpy array with the gain
    values, it should have the same shape as the
    `losses` array.
    :param losses: 1D numpy array with the loss
    values, it should have the same shape as the
    `gains` array.
    :param ox: 1D OX-values which bring every gain and loss
    value, it should have the same shape as the `losses`
    and `gains` arrays.
    :param xlim: Lineal space in which we will interpolate,
    specifically this will be an array with two components:
    the lower & upper bounds, respectively.
    :param round: Whether to round or not the chosen value.
    :param plot: Whether to plot or not the Gain
    VS Loss figure and its intersections, defaults to False.
    :param out_dir: Path at which we will save the figures.
    :param title: Figure's title.
    :param xlabel: OX label
    :param xtick_labels: Labels for every OX point.
    :param invert_xaxis: Whether to invert the OX axis or
    not (it will also change the intersection value chosen
    from the first to the last), default to False.

    :return float: The value at which both curves
    intersect.
    '''
    def fun(x, y): return (x+y)/2.0

    plt.rcParams['xtick.labelsize'] = FONT_SIZE/2.5
    plt.rcParams['ytick.labelsize'] = FONT_SIZE/2.5

    x_max, y_max = find_shared_maximum(
        ox, ox, gains,
        1-losses, xlim,
        fun=fun
    )

    if plot:
        plt.figure(figsize=(20, 5))

    t_x = np.round(x_max)\
        if round else x_max

    # Create a plot to see the curves
    if plot:
        plt.plot(
            ox, fun(gains, 1-losses),
            label=METRICS_FULL_NAMES["GAIN+CONS÷2"],
            color=METRIC_COLORS["GAIN+CONS÷2"]
        )
        plt.plot(
            ox, gains,
            label=METRICS_FULL_NAMES["GAIN"],
            color=METRIC_COLORS["GAIN"]
        )
        plt.plot(
            ox, 1-losses,
            label=METRICS_FULL_NAMES["CONS"],
            color=METRIC_COLORS["CONS"]
        )
        plt.vlines(
            t_x, 0, 1.1,
            linestyles='--',
            color="red",
            label="Valor escogido β"
        )

        plt.xticks(
            ox, labels=xtick_labels,
            rotation=45
        )

        plt.ylim((0, 1.1))
        plt.xlabel(
            xlabel,
            fontsize=FONT_SIZE
        )
        plt.ylabel(
            "Metric",
            fontsize=FONT_SIZE
        )

        if invert_xaxis:
            plt.gca().invert_xaxis()

        plt.grid()
        plt.legend(fontsize="x-large")

        plt.title(
            title,
            fontsize=FONT_SIZE
        )
        plt.tight_layout()
        save_plot(str(out_dir / title))

    return t_x
