from machine_learning import trees
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def base_tree():
    yield trees.BaseTree(eval_func=lambda y: y)


@pytest.fixture()
def n():
    yield 10


@pytest.fixture
def X(n):
    yield pd.DataFrame(
        {
            "cont_feature": np.arange(n),
            "binary_feature": [0] * (n // 2) + [1] * (n // 2),
            "multivalued_feature": np.repeat(np.arange(n // 2), 2),
        }
    )


@pytest.fixture()
def y(n):
    yield [0] * (n // 2) + [1] * (n // 2)


def test_get_unique_sorted():
    out = trees.BaseTree.get_unique_sorted(
        np.random.random_integers(low=0, high=10, size=100)
    )
    assert all(np.diff(out) > 0)
    assert len(np.unique(out)) == len(out)


@pytest.mark.parametrize("seed", list(np.arange(10)))
def test_get_midpoints(seed):
    np.random.seed(seed)
    arr = np.random.normal(size=100)
    out = trees.BaseTree.get_midpoints(arr)
    assert len(out) == len(arr) - 1
    for i in range(len(out)):
        assert out[i] == arr[i + 1] - (arr[i + 1] - arr[i]) / 2


def test_split_continuous_feature(X, y, base_tree):
    out = base_tree.get_optimal_continuous_feature_split(X, y, "cont_feature")


@pytest.mark.parametrize(
    "arr, y, expected",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], [1.5, 2.5, 3.5]),
        ([1, 1, 2, 3], [1, 2, 3, 4], [1.5, 2.5]),
        ([1, 1, 2, 3], [1, 2, 3, 3], [1.5]),
    ],
)
def test_get_valid_midpoints(arr, y, expected):
    assert np.allclose(
        trees.BaseTree.get_valid_midpoints(np.array(arr), np.array(y)), expected
    )


@pytest.fixture()
def leaf_node():
    yield trees.TreeSplits(children=[1, 2, 1, 2])


@pytest.fixture()
def root_node(leaf_node):
    yield trees.TreeSplits(nodes={"a": leaf_node, "b": leaf_node})


def test_collect_children(root_node):
    out = trees.BaseTree.collect_children(root_node)

    assert out.shape == (8,)
