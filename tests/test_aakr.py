"""Module for testing AAKR."""


import pytest

from sklearn.datasets import load_linnerud
from sklearn.utils.testing import assert_allclose

from aakr import AAKR


@pytest.fixture
def data():
    return load_linnerud(return_X_y=True)


def test_aakr(data):
    X = data[0]
    aakr = AAKR()
    assert aakr.metric == 'euclidean'
    assert aakr.bw == 1
    assert not aakr.modified
    assert aakr.penalty is None
    assert aakr.n_jobs == -1

    aakr.fit(X)
    assert hasattr(aakr, 'X_')

    X_nc = aakr.transform(X[:3])
    assert_allclose(X_nc, X[:3])


def test_aakr_fit_input_shape_mismatch(data):
    X = data[0]
    aakr = AAKR().fit(X)
    assert aakr.X_.shape[1] == X.shape[1]

    with pytest.raises(ValueError, match='Shape of input is different'):
        aakr.transform(X[:3, :-1])


def test_aakr_partial_fit_input_shape_mismatch(data):
    X = data[0]
    aakr = AAKR().partial_fit(X)
    assert aakr.X_.shape[1] == X.shape[1]

    with pytest.raises(ValueError, match='Shape of input is different'):
        aakr.partial_fit(X[:, :-1])


def test_aakr_modified(data):
    X = data[0]

    # Modified, no penalty given
    aakr = AAKR(modified=True, penalty=None)
    X_nc = aakr.fit(X).transform(X[:3])
    assert hasattr(aakr, 'X_')
    assert_allclose(X_nc, X[:3], atol=1.)

    # Modified, penalty given
    aakr = AAKR(modified=True, penalty=[1] * X.shape[1])
    X_nc = aakr.fit(X).transform(X[:3])
    assert hasattr(aakr, 'X_')
    assert_allclose(X_nc, X[:3], atol=1.)

    # Modified, penalty given, mismatch with input data
    with pytest.raises(ValueError, match='Shape of input is different from'):
        AAKR(modified=True, penalty=[1] * (X.shape[1] - 1)).fit(X)

    # No modified, penalty given
    with pytest.raises(ValueError, match='Parameter `penalty` given, but'):
        AAKR(modified=False, penalty=[1] * X.shape[1]).fit(X)
