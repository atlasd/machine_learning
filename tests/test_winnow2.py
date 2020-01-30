from machine_learning import winnow2
import pytest
import numpy as np
from itertools import product


@pytest.fixture
def X():
    yield np.array(list(product([0, 1], [0, 1], [0, 1])))


@pytest.fixture
def y(X):
    yield (X[:, 0] | X[:, 1]) & X[:, 2]


@pytest.fixture
def winnow2_model():
    yield winnow2.Winnow2(weight_scaler=2, threshold=2, num_features=3)


def test_predict(winnow2_model):
    out = winnow2_model.predict([1, 1, 1])
    assert out
    out = winnow2_model.predict([0, 0, 1])
    assert not out


def test_adjust_weights(winnow2_model):
    out = winnow2_model.adjust_weights(
        np.array([1.0, 1.0, 1.0]), scale_func=np.multiply
    )
    assert np.array_equal(out, np.array([2.0, 2.0, 2.0]))


@pytest.mark.parametrize(
    "X, y, weights",
    [
        (np.array([1, 1, 1]), 1, np.array([1, 1, 1])),
        (np.array([1, 1, 1]), 0, np.array([0.5, 0.5, 0.5])),
        (np.array([0.0, 0.0, 1.0]), 1, np.array([1.0, 1.0, 2.0])),
    ],
)
def test_run_training_iteration(winnow2_model, X, y, weights):
    winnow2_model.run_training_iteration(X=X, y=y)
    assert np.array_equal(winnow2_model.weights, weights)


def test_fit(X, y):
    model = winnow2.Winnow2(weight_scaler=2, threshold=0.5, num_features=3)
    model.fit(X, y)
    assert np.array_equal(model.weights, [0.25, 0.25, 0.5])
    assert np.array_equal(model.predict(X), y)
