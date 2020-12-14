"""Module for Auto Associative Kernel Regression models."""


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted


class AAKR(TransformerMixin, BaseEstimator):
    """Auto Associative Kernel Regressor.

    Please see the :ref:`Getting started <readme.rst>` documentation for more
        information.

    Parameters
    ----------
    metric : str, default='euclidean'
        Metric for calculating kernel distances.
    bw : float, default=1.0
        Kernel bandwith parameter.
    n_jobs : int, default=-1
        The number of jobs to run in parallel.

    Examples
    --------
    >>> from aakr import AAKR
    >>> import numpy as np
    >>> X = np.arange(100).reshape(50, 2)
    >>> aakr = AAKR()
    >>> aakr.fit(X)
    AAKR(metric='euclidean', bw=1, n_jobs=-1)
    """
    def __init__(self, metric='euclidean', bw=1, n_jobs=-1):
        self.metric = metric
        self.bw = bw
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit normal condition examples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training examples from normal conditions.
        y : None
            Not required, exists only for compability purposes.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validation
        X = check_array(X)

        # Save history
        self.X_ = X

        return self

    def partial_fit(self, X, y=None):
        """Fit more normal condition examples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training examples from normal conditions.
        y : None
            Not required, exists only for compability purposes.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validation
        X = check_array(X)

        if self.X_.shape[1] != X.shape[1]:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # Add new examples
        self.X_ = np.vstack((self.X_, X))

        return self

    def transform(self, X, **kwargs):
        """Transform given array into expected values in normal conditions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_nc : ndarray of shape (n_samples, n_features)
            Expected values in normal conditions for each sample and feature.
        """
        # Validation
        check_is_fitted(self, 'X_')

        X = check_array(X)

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # Kernel regression
        D = pairwise_distances(X=self.X_, Y=X, metric=self.metric,
                               n_jobs=self.n_jobs, **kwargs)
        k = 1 / np.sqrt(2 * np.pi * self.bw ** 2)
        w = k * np.exp(-D ** 2 / (2 * self.bw ** 2))
        X_nc = w.T.dot(self.X_) / w.sum(0)[:, None]

        return X_nc
