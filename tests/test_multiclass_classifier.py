import pytest
import numpy as np
from machine_learning import multiclass_classifier


@pytest.fixture
def mock_model_cls():
    class Model:
        def __init__(self, model_num, *args, **kwargs):
            self.model_num = model_num

        def fit(self, X, y):
            return

        def predict_prob(self, X):
            if self.model_num == 0:
                return [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
            if self.model_num == 1:
                return [1, 1, 1, 0, 0, 0, 2, 2, 2, 2]
            if self.model_num == 2:
                return [1, 1, 1, 2, 2, 2, 0, 0, 0, 0]

    yield Model


@pytest.fixture
def multiclass_c(mock_model_cls):
    yield multiclass_classifier.MulticlassClassifier(
        model_cls=mock_model_cls,
        classes=[0, 1, 2],
        cls_kwargs={i: {"model_num": i} for i in [0, 1, 2]},
    )


@pytest.fixture
def X():
    yield np.ones((10, 2))


@pytest.fixture
def y():
    yield np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])


def test_fit_raises(multiclass_c, X, y):
    y[0] = 3
    with pytest.raises(ValueError):
        multiclass_c.fit(X, y)


@pytest.mark.parametrize(
    "cls, expected",
    [
        (0, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
        (1, [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
        (2, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
    ],
)
def test_get_y_binary(y, cls, expected):
    out = multiclass_classifier.MulticlassClassifier._get_y_binary(y, cls)
    assert np.allclose(out, expected)


def test_predict(multiclass_c, X):
    out = multiclass_c.predict(X=X)
    assert np.allclose(out, [1, 1, 1, 2, 2, 2, 0, 0, 0, 0])
