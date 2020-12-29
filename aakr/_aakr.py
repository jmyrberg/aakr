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
        Gaussian Radial Basis Function (RBF) bandwith parameter.
    modified : bool, default=False
        Whether to use the modified version of AAKR (see reference [2]). The
        modified version reduces the contribution provided by those signals
        which are expected to be subject to the abnormal conditions.
    penalty : array-like or list of shape (n_features, 1) or None, default=None
            Penalty vector for the modified AAKR - only used when parameter
            modified=True. If modified AAKR used and penalty=None, penalty
            vector is automatically determined.
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
    def __init__(self, metric='euclidean', bw=1., modified=False, penalty=None,
                 n_jobs=-1):
        self.metric = metric
        self.bw = bw
        self.modified = modified
        self.penalty = penalty
        self.n_jobs = n_jobs

    def _fit_validation(self, X):
        X = check_array(X)

        if self.modified:
            if self.penalty is not None:
                penalty = check_array(self.penalty, ensure_2d=False)
                if len(penalty) != X.shape[1]:
                    raise ValueError('Shape of input is different from what '
                                     'is defined in penalty vector ('
                                     f'{X.shape[1]} != {len(penalty)})')
        elif not self.modified and self.penalty is not None:
            raise ValueError('Parameter `penalty` given, but `modified=False`.'
                             'Please set `modified=True` to make use of the '
                             'penalty vector, or set `penalty=None`.')

    def _rbf_kernel(self, X_obs_nc, X_obs):
        # Kernel regression
        D = pairwise_distances(X=X_obs_nc, Y=X_obs,
                               metric=self.metric, n_jobs=self.n_jobs)
        k = 1 / np.sqrt(2 * np.pi * self.bw ** 2)
        w = k * np.exp(-D ** 2 / (2 * self.bw ** 2))

        return w

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
        self._fit_validation(X)

        # Fit = save history
        # TODO: Add pruning options as a parameter... sampling?
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
        self._fit_validation(X)

        # Fit
        if hasattr(self, 'X_'):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError('Shape of input is different from what was '
                                 'seen in `fit` or `partial_fit`')
            self.X_ = np.vstack((self.X_, X))
        else:
            self.X_ = X

        return self

    def transform(self, X):
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

        # Modified AAKR basically sorts the columns
        # TODO: Needs to be verified that everything here is correct
        if self.modified:
            X_obs_nc = self.X_
            X_nc = np.zeros(X.shape)

            # Penalty matrix (J x J, where J is the number of features)
            if self.penalty is None:
                D = np.diag(np.arange(X.shape[1]) + 1) ** 2.
                D /= D.sum()
            else:
                D = np.diag(self.penalty).astype('float')

            for i, X_obs in enumerate(X):  # TODO: Vectorize
                # Standardized contributions in decreasing order (J, 1)
                diff = (np.abs(X_obs - X_obs_nc) / X_obs_nc.std(0)).sum(0)
                order = diff.argsort()[::-1]

                # Historical examples with ordered signals and penalty applied
                # (N_obs_nc x J)
                row_selector = np.arange(len(X_obs_nc))[:, np.newaxis]
                X_obs_nc_new = X_obs_nc[row_selector, order].dot(D)

                # New observations with ordered features and penalty applied
                # (1 x J)
                X_obs_new = X_obs[order].dot(D)[np.newaxis, :]

                # Weights for each observation (N_obs_nc, 1)
                w = self._rbf_kernel(X_obs_nc_new, X_obs_new)

                # Apply kernel and save the results (1, J)
                w_sum = w.sum(0)
                w_div = np.where(w_sum == 0, 1, w_sum)[:, np.newaxis]

                X_nc[i, :] = w.T.dot(X_obs_nc) / w_div
        else:
            w = self._rbf_kernel(self.X_, X)
            w_sum = w.sum(0)
            w_div = np.where(w_sum == 0, 1, w_sum)[:, np.newaxis]

            X_nc = w.T.dot(self.X_) / w_div

        return X_nc
