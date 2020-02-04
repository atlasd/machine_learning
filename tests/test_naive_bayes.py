import pytest
import numpy as np
from machine_learning import naive_bayes


@pytest.fixture
def model():
    yield naive_bayes.NaiveBayes(column_distribution_map={0: "multinomial"})


def test_fit_multinomial(model):
    X = np.array([0, 1, 2, 0, 1, 2]).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1])
    out = model._fit_multinomial(X, col_idx=0, y=y)
    assert np.allclose(out[0].p, [1 / 3, 1 / 3, 1 / 3])
