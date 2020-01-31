import pytest
import numpy as np
from machine_learning import validation


@pytest.fixture
def X():
    yield np.ones((100, 10))


@pytest.mark.parametrize(
    "num_folds, shuffle, expected",
    [(10, False, np.arange(100)), (10, True, np.arange(100, 0, -1)),],
)
def test_get_indices(X, num_folds, shuffle, expected, monkeypatch):
    monkeypatch.setattr(
        np.random, "shuffle", lambda *args, **kwargs: np.arange(100, 0, -1)
    )
    assert np.array_equal(
        validation.KFoldCV(num_folds=num_folds, shuffle=shuffle).get_indices(X=X),
        expected,
    )


@pytest.mark.parametrize(
    "indices, num_folds, expected",
    [
        (np.arange(10), 2, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (np.arange(10), 3, [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ],
)
def test_get_indices_split(indices, num_folds, expected):
    out = validation.KFoldCV._get_indices_split(indices, num_folds=num_folds)
    for out_arr, expt_arr in zip(out, expected):
        assert np.allclose(out_arr, expt_arr)


@pytest.mark.parametrize(
    "indices, num_split, expected",
    [
        ([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], 0, ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])),
        ([[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]], 0, ([0, 1, 2, 3], [4, 5, 6, 7, 8, 9])),
    ],
)
def test_get_one_split(indices, num_split, expected):
    train, test = validation.KFoldCV._get_one_split(indices, num_split=num_split)
    assert np.allclose(train, expected[1])
    assert np.allclose(test, expected[0])
