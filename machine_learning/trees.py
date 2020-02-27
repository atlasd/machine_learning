import pandas as pd
import numpy as np
from toolz import pipe
from collections import Counter


def uncertainty(y):
    return -1 * sum(
        [
            pipe(np.sum(y == value) / len(y), lambda ratio: ratio * np.log(ratio))
            for value in set(y)
        ]
    )


class TreeSplits:
    def __init__(
        self,
        feature_col=None,
        feature_value=None,
        node_type=None,
        nodes=None,
        children=[],
    ):
        self.feature_col = feature_col
        self.feature_value = feature_value
        self.node_type = node_type
        self.nodes = nodes
        self.children = children

    def update(self, feature_col, feature_value, node_type, nodes, children=[]):
        self.feature_col = feature_col
        self.feature_value = feature_value
        self.node_type = node_type
        self.nodes = nodes
        self.children = children


class BaseTree:
    def __init__(self, col_type_map, eval_func, agg_func, early_stopping_value=None):
        self.agg_func = agg_func
        self.col_type_map = col_type_map
        self.eval_func = eval_func
        self.early_stopping_value = (
            0 if not early_stopping_value else early_stopping_value
        )
        self.n_nodes = 1

    @staticmethod
    def get_unique_sorted(arr):
        return np.sort(np.unique(arr))

    @staticmethod
    def get_midpoints(arr):
        return arr[1:] - np.diff(arr) / 2

    @staticmethod
    def get_split_goodness_fit_continuous(arr, y, split, eval_func):
        above = arr >= split
        below = arr < split
        above_eval = eval_func(y[above]) * np.sum(above)
        below_eval = eval_func(y[below]) * np.sum(below)
        return above_eval + below_eval

    @staticmethod
    def get_min_across_splits_continuous(arr, y, splits, eval_func):
        post_split_evals = {
            idx: BaseTree.get_split_goodness_fit_continuous(
                arr=arr, y=y, split=split, eval_func=eval_func
            )
            for idx, split in enumerate(splits)
        }
        if not post_split_evals:
            import ipdb

            ipdb.set_trace()
        min_eval = min(post_split_evals, key=post_split_evals.get)
        return (splits[min_eval], post_split_evals.get(min_eval))

    def get_optimal_continuous_feature_split(self, X, y, feature_col):
        sorted_feature = BaseTree.get_unique_sorted(X[:, feature_col])
        midpoints = BaseTree.get_midpoints(sorted_feature)
        return BaseTree.get_min_across_splits_continuous(
            arr=X[:, feature_col], y=y, splits=midpoints, eval_func=self.eval_func
        )

    @staticmethod
    def get_discrete_split_value(arr, y, eval_func):
        return sum(
            [eval_func(y[arr == value]) * np.sum(arr == value) for value in set(arr)]
        )

    def get_optimal_discrete_feature_split(self, X, y, feature_col):
        return BaseTree.get_discrete_split_value(
            X[:, feature_col], y, eval_func=self.eval_func
        )

    def get_terminal_node(self, feature_col, node, feature_value, X, y):
        node_type = self.col_type_map[feature_col]
        if node_type == "continuous":
            above = X[:, feature_col] > feature_value
            node.update(
                feature_col=feature_col,
                feature_value=feature_value,
                node_type=node_type,
                nodes={
                    "above": TreeSplits(children=y[above]),
                    "below": TreeSplits(children=y[np.bitwise_not(above)]),
                },
            )
            self.n_nodes += 2
        else:
            unique_x_vals = set(X[:, feature_col])
            node.update(
                feature_col=feature_col,
                feature_value=None,
                nodes={
                    xval: TreeSplits(children=y[X[:, feature_col] == xval])
                    for xval in unique_x_vals
                },
                node_type="discrete",
            )
            self.n_nodes += len(unique_x_vals)

    def get_continuous_node(self, feature_col, feature_value, X, y, node):
        node.update(
            feature_col=feature_col,
            feature_value=feature_value,
            nodes={"below": TreeSplits(), "above": TreeSplits()},
            node_type="continuous",
        )

        above = X[:, feature_col] >= feature_value
        self.get_next_split(X=X[above], y=y[above], tree_split=node.nodes["above"])
        self.n_nodes += 1
        self.get_next_split(
            X=X[np.bitwise_not(above)],
            y=y[np.bitwise_not(above)],
            tree_split=node.nodes["below"],
        )
        self.n_nodes += 1

        return node

    def get_next_split(self, X, y, tree_split):
        column_values = {}

        if len(set(y)) == 1:
            tree_split.update(
                feature_col=None,
                feature_value=None,
                node_type=None,
                nodes={},
                children=y,
            )
            return tree_split

        for k, v in self.col_type_map.items():
            if len(set(X[:, k])) == 1:
                value = np.inf
                split = None
            elif v == "continuous":
                split, value = self.get_optimal_continuous_feature_split(
                    X=X, y=y, feature_col=k
                )
            else:
                value = self.get_optimal_discrete_feature_split(X=X, y=y, feature_col=k)
                split = None

            column_values[k] = (split, value)

        col_idx_with_min_value = min(
            column_values, key=lambda x: column_values.get(x)[1]
        )

        if (
            column_values.get(col_idx_with_min_value)[1] <= self.early_stopping_value
        ) or not np.isfinite(column_values.get(col_idx_with_min_value)[1]):
            self.get_terminal_node(
                feature_col=col_idx_with_min_value,
                feature_value=column_values[col_idx_with_min_value][0],
                node=tree_split,
                X=X,
                y=y,
            )
            return tree_split

        if self.col_type_map.get(col_idx_with_min_value) == "continuous":
            return self.get_continuous_node(
                feature_col=col_idx_with_min_value,
                feature_value=column_values[col_idx_with_min_value][0],
                X=X,
                y=y,
                node=tree_split,
            )

        else:
            unique_x_vals = set(X[:, col_idx_with_min_value])
            tree_split.update(
                feature_col=col_idx_with_min_value,
                feature_value=column_values[col_idx_with_min_value][0],
                nodes={xval: TreeSplits() for xval in unique_x_vals},
                node_type="discrete",
            )

            for x_col_value in unique_x_vals:
                matches = X[:, col_idx_with_min_value] == x_col_value
                self.get_next_split(
                    X=X[matches], y=y[matches], tree_split=tree_split.nodes[x_col_value]
                )
                self.n_nodes += 1

            return tree_split

    def fit(self, X, y):
        self.root = TreeSplits()
        self.get_next_split(X=X, y=y, tree_split=self.root)

    def predict_node(self, X, node):
        if node.children is not None and len(node.children):
            return self.agg_func(node.children)

        if node.node_type == "continuous":
            if X[node.feature_col] > node.feature_value:
                return self.predict_node(X=X, node=node.nodes["above"])
            else:
                return self.predict_node(X=X, node=node.nodes["below"])

        return self.predict_node(X=X, node=node.nodes[X[node.feature_col]])

    def predict(self, X):
        if X.ndim == 1:
            return self.predict_node(X=X, node=self.root)

        return [self.predict_node(X=row, node=self.root) for row in X]


class DecisionTreeClassifier(BaseTree):
    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: Counter(y).most_common(1)[0][0],
        )


class DecisionTreeRegressor(BaseTree):
    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: np.mean(y),
        )
