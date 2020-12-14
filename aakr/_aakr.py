"""Module for models."""


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted


class AAKR(TransformerMixin, BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from aakr import AAKR
    >>> import numpy as np
    >>> X = np.arange(100).reshape(50, 2)
    >>> aakr = AAKR()
    >>> aakr.fit(X)
    AAKR(metric='euclidean', bw=1, n_jobs=None)
    """
    def __init__(self, metric='euclidean', bw=1, n_jobs=None):
        self.metric = metric
        self.bw = bw
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training examples from normal conditions.
        y : None
            Not needed, exists for compability purposes.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self.X_ = X
        return self

    def predict(self, X, **kwargs):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_nc : ndarray, shape (n_samples, n_features)
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
