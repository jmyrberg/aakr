"""Module for Auto Associative Kernel Regression models."""


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted


class AAKR(TransformerMixin, BaseEstimator):
    """Auto Associative Kernel Regression.

    Parameters
    ----------
    metric : str, default='euclidean'
        Metric for calculating kernel distances, see available metrics from
        `sklearn.metrics.pairwise_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html>`_.
    bw : float, default=1.0
        Kernel bandwith parameter.
    n_jobs : int, default=-1
        The number of jobs to run in parallel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Historical normal condition examples given as an array.

    References
    ----------
    .. [1] Chevalier  R., Provost  D., and Seraoui R., 2009,
           “Assessment of Statistical and Classification Models For Monitoring
           EDF’s  Assets”,  Sixth  American  Nuclear  Society  International
           Topical Meeting on Nuclear Plant Instrumentation.
    .. [2] Baraldi P., Di Maio F., Turati P., Zio E., 2014,
           "A modified Auto Associative Kernel Regression method for robust
           signal reconstruction in nuclear power plant components", European
           Safety and Reliability Conference ESREL.
    """
    def __init__(self, metric='euclidean', bw=1, n_jobs=-1):
        self.metric = metric
        self.bw = bw
        self.n_jobs = n_jobs
        # TODO: Implement modified -version

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

        # Fit
        if hasattr(self, 'X_'):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError('Shape of input is different from what was '
                                 'seen in `fit` or `partial_fit`')
            self.X_ = np.vstack((self.X_, X))
        else:
            self.X_ = X

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
        w_sum = w.sum(0)
        X_nc = w.T.dot(self.X_) / np.where(w_sum == 0, 1, w_sum)[:, None]

        return X_nc
