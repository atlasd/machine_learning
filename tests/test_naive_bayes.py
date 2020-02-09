import pytest
import numpy as np
from machine_learning import naive_bayes
from scipy import stats


@pytest.fixture
def model():
    yield naive_bayes.NaiveBayes(
        column_distribution_map={0: "multinomial", 1: "gaussian"}
    )


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


@pytest.fixture
def X():
    yield np.array([[0, 1, 2, 0, 1, 2], [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]]).T


@pytest.fixture
def y():
    yield np.array([0, 0, 0, 1, 1, 1])


@pytest.fixture
def fitted_model(X, y, model):
    model.fit(X, y)
    yield model


def test_fit(fitted_model):
    assert np.allclose(
        fitted_model.fitted_distributions[0][0].p, np.array([1 / 3, 1 / 3, 1 / 3])
    )
    assert np.allclose(
        fitted_model.fitted_distributions[0][1].p, np.array([1 / 3, 1 / 3, 1 / 3])
    )
    assert np.isclose(fitted_model.fitted_distributions[1][0].mean(), 0.2 / 3)
    assert np.isclose(fitted_model.fitted_distributions[1][1].mean(), -0.2 / 3)

    assert np.isclose(
        fitted_model.fitted_distributions[1][0].std(), np.std([0.1, -0.1, 0.2])
    )
    assert np.isclose(
        fitted_model.fitted_distributions[1][1].std(), np.std([0.3, -0.3, 0.2])
    )

    assert np.allclose(fitted_model.prior.p, [0.5, 0.5])


def test_predict_one_class(fitted_model, X):
    out0 = fitted_model._predict_one_class(X, class_idx=0)
    out1 = fitted_model._predict_one_class(X, class_idx=1)
    assert out0.shape[0] == X.shape[0]
    assert out1.shape[0] == X.shape[0]


def test_predict(fitted_model, X):
    out = fitted_model.predict(X=X)
    assert out.shape[0] == X.shape[0]
    assert out.ndim == 1
    assert np.allclose(out, [0, 1, 0, 1, 1, 1])
