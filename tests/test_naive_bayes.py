import pytest
import numpy as np
from machine_learning import naive_bayes


@pytest.fixture
def model():
    yield naive_bayes.NaiveBayes(column_distribution_map={0: "multinomial"})


@pytest.mark.parametrize(
    "X, y, expected_p, expected_n, expected_p1",
    [
        (
            np.array([0, 1, 2, 0, 1, 2]).reshape(-1, 1),
            np.array([0, 0, 0, 1, 1, 1]),
            [1 / 3, 1 / 3, 1 / 3],
            3,
            [1 / 3, 1 / 3, 1 / 3],
        ),
        (
            np.array([0, 1, 1, 0, 1, 2]).reshape(-1, 1),
            np.array([0, 0, 0, 1, 1, 1]),
            [2 / 6, 3 / 6, 1 / 6],
            3,
            [1 / 3, 1 / 3, 1 / 3],
        ),
    ],
)
def test_fit_multinomial(model, X, y, expected_p, expected_n, expected_p1):
    out = model._fit_multinomial(X, col_idx=0, y=y)
    assert np.allclose(out[0].p, expected_p)
    assert out[0].n == expected_n
    assert np.allclose(out[1].p, expected_p1)


def test_fit_gaussian(model):
    X = np.array([-1, 1, -2, 2]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    out = model._fit_gaussian(X, 0, y)
    assert out[0].kwds["loc"] == 0
    assert out[1].kwds["loc"] == 0
    assert np.isclose(out[0].kwds["scale"], 1)
    assert np.isclose(out[1].kwds["scale"], 2)
