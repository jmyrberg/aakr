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
    assert aakr.n_jobs is None

    aakr.fit(X)
    assert hasattr(aakr, 'X_')

    X_nc = aakr.predict(X[:3])
    assert_allclose(X_nc, X[:3])


def test_aakr_input_shape_mismatch(data):
    X = data[0]
    aakr = AAKR().fit(X)
    assert aakr.X_.shape[1] == X.shape[1]

    with pytest.raises(ValueError, match='Shape of input is different'):
        aakr.predict(X[:3, :-1])
