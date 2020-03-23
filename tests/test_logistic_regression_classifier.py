from machine_learning import linear
import numpy as np
import pytest


def test_matrix_product():
    out = linear.LogisticRegressionClassifier.get_matrix_product(
        X=np.ones((3, 4)), weights=np.ones((2, 4))
    )

    assert out.shape == (2, 3)


@pytest.mark.parametrize(
    "X_shape, weights_shape, expected_shape", [((10, 4), (3, 4), (10, 3))]
)
def test_get_class_scores(X_shape, weights_shape, expected_shape):
    out = linear.LogisticRegressionClassifier.get_class_scores(
        X=np.ones(X_shape), weights=np.ones(weights_shape)
    )

    assert out.shape == expected_shape
    assert np.allclose(out[:, 0], out[:, 1])
    assert np.allclose(out[:, 0], out[:, 2])


@pytest.mark.parametrize(
    "X, y, weights", [(np.ones((10, 3)), np.array([0] * 5 + [1] * 5), np.ones((2, 3)))]
)
def test_get_gradient_update_term(X, y, weights):
    out = linear.LogisticRegressionClassifier.get_gradient_update_term(
        X=X, y=y, weights=weights, alpha=0
    )
    assert out.shape == weights.shape


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1], [0]]), np.array([0, 1,])),
        (np.array([[1, 0], [0, 0]]), np.array([0, 1])),
    ],
)
def test_fit_predict(X, y):
    lrc = linear.LogisticRegressionClassifier(2)
    lrc.fit(X, y)
    out = lrc.predict(X)
    assert np.allclose(out, y)
