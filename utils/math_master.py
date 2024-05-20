import numpy as np


def find_intersections(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        xlim: np.ndarray,
        n: int = 1000) -> np.ndarray:
    '''
    This method will find the intersection between
    two numpy arrays by performing linear interpolation.

    For more information see `numpy.interp`.

    To know how do we obtain the intersection between
    two discrete arrays see:
    https://stackoverflow.com/questions/28766692/how-to-find-the-intersection-of-two-graphs

    Example:
    ```
    _min = -5
    _max = 5

    ox = np.linspace(
        _min, _max, 20)
    y1 = np.sin(ox)
    y2 = np.cos(ox)

    ox_int, oy_int = find_intersections(
        ox, ox, y1, y2, [_min, _max])

    plt.plot(ox, y1)
    plt.plot(ox, y2)
    plt.scatter(
        ox_int, oy_int)

    plt.show()
    ```

    :param x1: 1D numpy array with the input values of
    the first function.
    :param x2: 1D numpy array with the input values of
    the second function.
    :param y1: 1D numpy array with the images of the
    first function.
    :param y2: 1D numpy array with the images of the
    second function.
    :param xlim: Lineal space in which we will interpolate,
    specifically this will be an array with two components:
    the lower & upper bounds, respectively.
    :param n: Number of samples to interpolate between
    the lower & upper bounds.

    :return Tuple[np.ndarray x2]: A tuple with two arrays:
    the first one contains the OX values on which both
    interpolations intersect, and the second one the OY
    values on which they intersect.
    '''
    ox = np.linspace(xlim[0], xlim[1], n)
    y1_interp = np.interp(ox, x1, y1)
    y2_interp = np.interp(ox, x2, y2)

    intersec_idxs = np.argwhere(
        np.diff(np.sign(
            y1_interp - y2_interp))
    ).flatten()

    return ox[intersec_idxs], y1_interp[intersec_idxs]


def find_shared_maximum(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        xlim: np.ndarray,
        fun: callable,
        n: int = 1000) -> np.ndarray:
    '''
    This method will find the values which maximise both
    of the arrays by performing linear interpolation and
    a specific operation defined by the user in `fun`.

    For more information see `numpy.interp`.

    Example:
    ```
    _min = -5
    _max = 5

    ox = np.linspace(
        _min, _max, 20)
    y1 = 2*ox
    y2 = -2*ox + 10

    ox_int, oy_int = find_shared_maximum(
        ox, ox, y1, y2, [_min, _max],
        fun=lambda x, y: x*y
    )

    plt.plot(ox, y1)
    plt.plot(ox, y2)
    plt.scatter(
        ox_int, oy_int)

    plt.show()
    ```

    :param x1: 1D numpy array with the input values of
    the first function.
    :param x2: 1D numpy array with the input values of
    the second function.
    :param y1: 1D numpy array with the images of the
    first function.
    :param y2: 1D numpy array with the images of the
    second function.
    :param xlim: Lineal space in which we will interpolate,
    specifically this will be an array with two components:
    the lower & upper bounds, respectively.
    :param fun: Function to maximise, this function
    should take two 1D numpy arrays and return another
    1D numpy array with the same shape as the inputs.
    This will be the function to maximise!
    :param n: Number of samples to interpolate between
    the lower & upper bounds.

    :return Tuple[intx2]: A tuple with two values:
    the first one is the OX value on which both
    interpolations are maximised, and the second
    one is the maximised OY value.
    '''
    ox = np.linspace(xlim[0], xlim[1], n)
    y1_interp = np.interp(ox, x1, y1)
    y2_interp = np.interp(ox, x2, y2)

    max_idx = np.argmax(
        fun(y1_interp, y2_interp))

    return ox[max_idx], y1_interp[max_idx]
