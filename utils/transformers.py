'''
This module will define some data
transformers compatible with SkLearn.

:Author: Javier Jiménez Rodríguez
(javier.jimenez02@estudiante.uam.es)
:Date: 15/03/2024
'''

from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.base import TransformerMixin, BaseEstimator


def choose_scaler(
        n_features: int,
        scaler: str = None) -> TransformerMixin:
    '''
    This method will return a scaler
    specified through arguments.

    :param n_features: Number of features, WARNING:
    this is not the combination of "n_feats x n_timesteps",
    but the number of multivariate time series itself! (i.e. `n_feats`)
    :param scaler: Name of the scaler, available
    options are:
    · "MinMax" for Min-Max scaling in [-1, 1].
    · "Standardize" for standardization.
    · None for no transformation at all (identity).

    :return TransformerMixin: The transformer specified.
    '''
    # Choose a time series scaler
    if scaler == "MinMax":
        scaler = SkLearnWrapper(
            n_features,
            TimeSeriesMinMaxScaler(
                feature_range=(-1, 1))
        )
    elif scaler == "Standardize":
        scaler = SkLearnWrapper(
            n_features,
            TimeSeriesStandardScaler()
        )
    else:
        scaler = SkLearnWrapper(
            n_features,
            TimeSeriesIdentityScaler()
        )

    return scaler


class SkLearnWrapper(TransformerMixin, BaseEstimator):
    '''
    This class will wrap SkLearn datasets with shape:
        (n_samples, n_features x n_timesteps)
    To work with time series transformers.
    '''

    def __init__(
            self,
            n_feats: int,
            transformer: TransformerMixin):
        self.n_feats = n_feats
        self.transformer = transformer

    def __wrap_data(self, X: np.ndarray) -> np.ndarray:
        '''
        This method wraps another given `transformer`
        transforming a 2D SkLearn array into
        3D time series.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats*n_timesteps)

        scaler = TimeSeriesMinMaxScaler(
            feature_range=(0, 1))
        wrapper = SkLearnWrapper(n_feats, scaler)

        wrapper.__wrap_data(X).shape
        ----- Output -----
        (100, 10, 32)
        ```

        :param X: 2D array with shape:
            (n_samples, n_feats x n_timesteps)
        :return np.ndarray: A 3D Numpy array with
        the time series and shape:
            (n_samples, n_feats, n_timesteps)
        '''
        n_samples, _ = X.shape
        X_ts = X.reshape(
            (n_samples, self.n_feats, -1))
        return X_ts

    def __unwrap_data(self, X: np.ndarray) -> np.ndarray:
        '''
        This method unwraps the output given by
        `self.transformer` transforming 3D time series
        into a 2D SkLearn array.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats, n_timesteps)

        scaler = TimeSeriesMinMaxScaler(
            feature_range=(0, 1))
        wrapper = SkLearnWrapper(n_feats, scaler)

        wrapper.unwrap_data(X).shape
        ----- Output -----
        (100, 320)
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :return np.ndarray: A 2D Numpy array with
        the shape:
            (n_samples, n_feats x n_timesteps)
        '''
        n_samples, _, _ = X.shape
        return X.reshape(n_samples, -1)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> SkLearnWrapper:
        '''
        A fit-wrapper for SkLearn's datasets.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats*n_timesteps)

        scaler = TimeSeriesMinMaxScaler(
            feature_range=(0, 1))
        wrapper = SkLearnWrapper(n_feats, scaler)

        X_scaled = wrapper.fit_transform(X)

        np.min(X_scaled), np.max(X_scaled), X.shape
        ----- Output -----
        (0.0, 1.0, (100, 320))
        ```

        :param X: 2D array with shape:
            (n_samples, n_feats x n_timesteps)
        :param y: Target array (unused), defaults to None.
        :return SkLearnWrapper: The same object.
        '''
        self.transformer = self.transformer.fit(
            self.__wrap_data(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''
        A transform-wrapper for SkLearn's datasets.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats*n_timesteps)

        scaler = TimeSeriesMinMaxScaler(
            feature_range=(0, 1))
        wrapper = SkLearnWrapper(n_feats, scaler)

        X_scaled = wrapper.fit_transform(X)

        np.min(X_scaled), np.max(X_scaled), X.shape
        ----- Output -----
        (0.0, 1.0, (100, 320))
        ```

        :param X: 2D array with shape:
            (n_samples, n_feats x n_timesteps)
        :return np.ndarray: The transformed 2D array.
        '''
        return self.__unwrap_data(
            self.transformer.transform(
                self.__wrap_data(X))
        )


class TimeSeriesMinMaxScaler(TransformerMixin, BaseEstimator):
    '''
    A time series min-max scaler with a different scaling
    pair (min & max) per feature.
    '''

    def __init__(self, feature_range: Tuple[int, int]):
        self.feature_range = feature_range

    def fit(
            self,
            X: np.array,
            y: np.array = None) -> TimeSeriesMinMaxScaler:
        '''
        This method will fit a MinMax scaler inspired from
        the SkLearn's page for time series.

        Example:
        ```
        np.random.seed(1234)

        X = np.random.randn(100, 10, 32)
        scaler = TimeSeriesMinMaxScaler(
            feature_range=(100, 200))
        X_scaled = scaler.fit_transform(X)

        np.min(X_scaled), np.max(X_scaled)
        ----- Output -----
        (100.0, 200.0)
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :param y: Target array (unused), defaults to None.
        :return TimeSeriesMinMaxScaler: The same object.
        '''
        X_flat = np.swapaxes(X, 0, 1).reshape(
            X.shape[1], -1)

        self.min_ = np.min(
            X_flat, axis=1).reshape(1, -1, 1)
        self.max_ = np.max(
            X_flat, axis=1).reshape(1, -1, 1)

        return self

    def transform(self, X: np.array) -> np.array:
        '''
        This method will scale the time series.

        Example:
        ```
        np.random.seed(1234)

        X = np.random.randn(100, 10, 32)
        scaler = TimeSeriesMinMaxScaler(
            feature_range=(100, 200))
        X_scaled = scaler.fit_transform(X)

        np.min(X_scaled), np.max(X_scaled)
        ----- Output -----
        (100.0, 200.0)
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :return np.array: Transformed array.
        '''
        return self.feature_range[0] + (X - self.min_) *\
            (self.feature_range[1] - self.feature_range[0]) /\
            (self.max_ - self.min_)


class TimeSeriesStandardScaler(TransformerMixin, BaseEstimator):
    '''
    A time series standard scaler with a different scaling
    pair (mean & standard deviation) per feature.
    '''

    def fit(
            self,
            X: np.array,
            y: np.array = None) -> TimeSeriesStandardScaler:
        '''
        This method will fit a Standard scaler inspired from
        the SkLearn's page for time series.

        Example:
        ```
        np.random.seed(1234)

        X = np.random.randn(100, 10, 32)
        scaler = TimeSeriesStandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.mean(X_scaled), np.std(X_scaled)
        ----- Output -----
        (4.440892098500626e-18, 1.0)
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :param y: Target array (unused), defaults to None.
        :return TimeSeriesStandardScaler: The same object.
        '''
        X_flat = np.swapaxes(X, 0, 1).reshape(
            X.shape[1], -1)

        self.mu_ = np.mean(
            X_flat, axis=1).reshape(1, -1, 1)
        self.std_ = np.std(
            X_flat, axis=1).reshape(1, -1, 1)

        return self

    def transform(self, X: np.array) -> np.array:
        '''
        This method will scale the time series.

        Example:
        ```
        np.random.seed(1234)

        X = np.random.randn(100, 10, 32)
        scaler = TimeSeriesStandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.mean(X_scaled), np.std(X_scaled)
        ----- Output -----
        (4.440892098500626e-18, 1.0)
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :return np.array: Transformed array.
        '''
        return (X - self.mu_)/self.std_


class TimeSeriesIdentityScaler(TransformerMixin, BaseEstimator):
    '''
    This class is created to avoid making
    any data transformation and work with
    the original data.
    '''

    def fit(self,
            X: np.array,
            y: np.array = None) -> TimeSeriesIdentityScaler:
        '''
        This method won't do anything.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats, n_timesteps)

        scaler = TimeSeriesIdentityScaler()
        np.all(X == scaler.fit_transform(X))
        ----- Output -----
        True
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :param y: Target array (unused), defaults to None.
        :return TimeSeriesIdentityScaler: The same object.
        '''
        return self

    def transform(self, X: np.array) -> np.array:
        '''
        This method will return the
        same input array.

        Example:
        ```
        np.random.seed(1234)

        n_samples = 100
        n_feats = 10
        n_timesteps = 32

        X = np.random.randn(
            n_samples, n_feats, n_timesteps)

        scaler = TimeSeriesIdentityScaler()
        np.all(X == scaler.fit_transform(X))
        ----- Output -----
        True
        ```

        :param X: 3D array with shape:
            (n_samples, n_feats, n_timesteps)
        :return np.array: Original array.
        '''
        return X
