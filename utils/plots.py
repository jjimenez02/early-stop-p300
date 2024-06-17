import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from typing import List, Tuple, Dict, Union
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def save_plot(fig_path: str):
    '''
    This method will save an already
    loaded plot into the specified path.

    Example:
    ```
    plt.scatter(1, 1)
    save_plot("lmao/hey.png")
    ```

    :param fig_path: Desired path for the
    figure.
    '''
    os.makedirs(
        os.path.dirname(fig_path),
        exist_ok=True
    )

    plt.savefig(fig_path)
    plt.clf()


def stim_power_bar_plot(
        names: List[str],
        values: np.array,
        labels: List[str],
        xlabel: str = "",
        ylabel: str = "",
        ymin: float = None,
        ymax: float = None,
        title: str = ""):
    '''
    This method will plot the stimulus
    power for a series of values given.

    Example:
    ```
    x = [1, 2, 3]
    y = [1, .5, 2]
    labels = ["A1", "B2", "C3"]

    stim_power_bar_plot(x, y, labels)
    ```

    :param names: Name/number of the stimulus
    to associate with the values (a list).
    :param values: Power of the stimulus
    to associate with the names (a list).
    :param labels: Stimulus labels to print and
    associate with the values (a list).
    :param xlabel: OX label, defaults to ""
    :param ylabel: OY label, defaults to ""
    :param ymin: OY minimum value to show, defaults
    to None.
    :param ymax: OY maximum value to show, defaults
    to None.
    '''
    min_val = np.min(values)
    max_val = np.max(values)

    plt.bar(
        names, values,
        label=labels
    )

    plt.axhline(
        min_val, linestyle='solid',
        color="red", linewidth=.5
    )
    plt.axhline(
        max_val, linestyle='solid',
        color="green", linewidth=.5
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.xticks(names)

    if ymin is None:
        plt.ylim(ymax=ymax)
    else:
        plt.ylim(ymin - ymin/10, ymax)

    plt.grid()
    plt.tight_layout()


def plot_accuracy_by_trial(
        scores: np.array,
        title: str = "Accuracy",
        ylim: Tuple[int, int] = (None, None),
        label: str = None,
        ylabel: str = "Score"):
    '''
    This method will plot the accuracy
    by trial achieved by the classifier.

    Example:
    ```
    plot_accuracy_by_trial([.3, .6, .8])
    ```

    :param scores: Array with all the
    scores per trial.
    :param title: Plot title.
    :param ylim: OY limits, defaults to
    (None, None).
    :param label: Curve's label.
    '''
    x = np.arange(1, len(scores) + 1)
    y = scores

    plt.plot(x, y, label=label)

    plt.xlabel("Trials")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(x, rotation=45)
    plt.ylim(ylim)
    plt.grid(visible=True)

    plt.tight_layout()


def scores_bar_plot(
        names: List[str],
        scores: Dict[str, List[float]],
        colors: Dict[str, str],
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        ymin: float = None,
        yerr: Dict[str, List[float]] = None,
        bars_width: float = .25,
        figsize: Tuple[int, int] = (10, 10),
        rotation: int = 45,
        ymax: float = 1,
        fontsize: float = 10,
        legend_fontsize: Union[float, str] = 'small') -> None:
    '''
    This method will plot a grouped-bar-plot,
    adapted from:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

    Example:
    ```
    np.random.seed(42)

    names = ['A', 'B', 'C']
    n_vals = len(names)

    values = {
        'D': np.random.rand(n_vals),
        'E': np.random.rand(n_vals),
        'F': np.random.rand(n_vals)
    }

    colors = {
        'D': 'r',
        'E': 'g',
        'F': 'b'
    }

    scores_bar_plot(
        names, values, colors)
    ```

    :param names: Techniques names.
    :param scores: Techniques scores.
    :param colors: Correspondences between the score
    name and its color.
    :param xlabel: OX label, defaults to ""
    :param ylabel: OY label, defaults to ""
    :param title: Plot title, defaults to ""
    :param ymin: Minimum value to show in order to plot
    all the bar-plots within the same scale, if it is `None`
    it will be inferred from data, defaults to None.
    :param yerr: Parameter with the standard deviations
    (same structure as `scores`), defaults to None.
    :param bars_width: Width of every bar, defaults to
    0.25.
    :param fig_size: Figure's size, defaults to (10, 10)
    :param rotation: XLabel's rotation.
    :param ymax: Maximum OY value to plot, defaults to 1.
    :param fontsize: Labels & title fontsize, defaults to 10.
    :param legend_fontsize: Legend fontsize, defaults to 'small'.
    '''
    x = np.arange(len(names))  # the label locations
    multiplier = 0

    fig, ax = plt.subplots(
        layout='constrained', figsize=figsize)

    global_min_val = np.inf
    for attribute in scores.keys():
        measurement = scores[attribute]
        deviations = yerr[attribute]\
            if yerr is not None else None

        min_val = np.min(measurement)
        if min_val < global_min_val:
            global_min_val = min_val

        offset = bars_width * multiplier
        ax.bar(
            x + offset,
            measurement,
            bars_width,
            label=attribute,
            linewidth=1,
            edgecolor="black",
            color=colors[attribute],
            yerr=deviations,
            capsize=5,
            ecolor="grey",
        )
        multiplier += 1

    if len(scores.keys()) == 1:
        ax.set_xticks(
            x,
            names,
            rotation=rotation
        )
    else:
        ax.set_xticks(
            x + bars_width,
            names,
            rotation=rotation
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    ax.legend(
        loc='lower left',
        framealpha=.6,
        fontsize=legend_fontsize
    )
    ax.grid(linestyle='dotted')

    if ymin is None:
        ax.set_ylim(
            global_min_val - 1e-3, ymax)
    else:
        ax.set_ylim(ymin - 1e-3, ymax)


def radar_factory(
        num_vars: int,
        frame: str = 'circle'):
    '''
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Note that this method has been adapted from:
    https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    Example:
    ```
    np.random.seed(1234)

    n_scores = 8
    n_methods = 3

    X = np.random.randn(
        n_scores, n_methods)

    theta = radar_factory(n_scores)

    fig, axs = plt.subplots(
        figsize=(9, 9), nrows=1, ncols=1,
        subplot_kw=dict(projection='radar'))
    for method_idx in range(n_methods):
        axs.plot(theta, X[:, method_idx])

    plt.show()
    ```

    :param num_vars: Number of variables for radar chart.
    :param frame: Shape of frame surrounding axes, possible values
    are {'circle', 'polygon'}, defaults to 'circle'.

    :return np.ndarray: An array with the evenly spaced angles
    at which we will plot each of the values.
    '''
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N', offset=22)

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_heatmap(
        X: np.array,
        xlabel: str,
        ylabel: str,
        title: str,
        xticks: List[str],
        yticks: List[str],
        figsize: Tuple[int, int] = (10, 10),
        vlim: Tuple[float, float] = (None, None)):
    '''
    This method will plot a heatmap considering
    three dimensions: features, timesteps and power.

    Example:
    ```
    np.random.seed(1234)

    n_feats = 10
    n_ts = 32
    X = np.random.randn(
        n_feats, n_ts)
    X[0] = 10

    plot_heatmap(
        X, "XLabel", "YLabel",
        "Title", np.arange(n_ts),
        np.arange(n_feats), (10, 10)
    )

    plt.show()
    print("DEBUG")
    ```

    :param X: 2D Numpy array with the values and shape:
        `(n_features, n_timesteps)`
    :param xlabel: OX label.
    :param ylabel: OY label.
    :param title: Title for the plot.
    :param xticks: OX's labels.
    :param yticks: Features's names.
    :param figsize: Figure's size.
    :param vlim: Colorbar with the range of values,
    defaults to `(None, None)`.
    '''
    _, n_timesteps = X.shape
    fig, axs = plt.subplots(
        nrows=1, ncols=1, figsize=figsize)

    im = axs.imshow(
        X, cmap="coolwarm",
        vmin=vlim[0], vmax=vlim[1]
    )

    divider = make_axes_locatable(axs)
    cax = divider.append_axes(
        "right", size="5%", pad=0.05)

    axs.figure.colorbar(
        im, ax=axs, cax=cax)

    axs.set_xticks(
        np.arange(len(xticks)),
        labels=[np.round(x, decimals=2)
                if i % 3 == 0 else ''
                for i, x in enumerate(xticks)],
        rotation=45
    )
    axs.set_yticks(
        np.arange(len(yticks)),
        labels=yticks,
        rotation=0
    )

    axs.set_title(title)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)

    fig.tight_layout()
