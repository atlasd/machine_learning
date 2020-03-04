import pandas as pd
import numpy as np
from toolz import pipe
from collections import Counter
import multiprocessing
import operator
import sys
import copy

sys.setrecursionlimit(10000)


def entropy(y):
    return -1 * sum(
        [
            pipe(np.sum(y == value) / len(y), lambda ratio: ratio * np.log(ratio))
            for value in set(y)
        ]
    )


def negative_mse(y):
    return -1 * mse(y)


def mse(y):
    return np.mean((y - np.mean(y)) ** 2)


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

    def is_leaf(self):
        return self.nodes is None or len(self.nodes) == 0


class BaseTree:
    def __init__(
        self,
        col_type_map,
        eval_func,
        agg_func,
        early_stopping_value=None,
        early_stopping_comparison=operator.le,
    ):
        self.agg_func = agg_func
        self.col_type_map = col_type_map
        self.eval_func = eval_func
        self.early_stopping_value = (
            0 if not early_stopping_value else early_stopping_value
        )
        self.n_nodes = 1
        self.early_stopping_comparison = early_stopping_comparison

    @staticmethod
    def get_valid_midpoints(arr, y):
        idxs = np.argsort(arr)
        sorted_arr = arr[idxs]
        arr_diffs = np.diff(sorted_arr)
        midpoints = sorted_arr[1:] - arr_diffs / 2
        valid_midpoints = midpoints[np.bitwise_and(arr_diffs > 0, np.diff(y[idxs]) > 0)]

        return valid_midpoints

        # return np.sort(np.unique(arr))

    @staticmethod
    def get_midpoints(arr):
        return arr[1:] - np.diff(arr) / 2

    @staticmethod
    def get_split_goodness_fit_continuous(arr, y, split, eval_func):
        above = arr >= split
        below = arr < split
        # get weighted average eval_func

        n_above = np.sum(above)

        above_eval = eval_func(y[above]) * n_above / len(y)
        below_eval = eval_func(y[below]) * (len(y) - n_above) / len(y)

        return (
            above_eval + below_eval,
            -1
            * sum(
                map(
                    lambda x: x * np.log(x),
                    [n_above / len(y), (len(y) - n_above) / len(y)],
                )
            ),
        )

    @staticmethod
    def get_min_across_splits_continuous(arr, y, splits, eval_func):
        n = len(splits)
        if n > 500:
            with multiprocessing.Pool(processes=8) as p:
                post_split_evals = dict(
                    zip(
                        range(len(splits)),
                        p.starmap(
                            BaseTree.get_split_goodness_fit_continuous,
                            zip([arr] * n, [y] * n, splits, [eval_func] * n),
                        ),
                    )
                )
                p.close()
        else:
            post_split_evals = dict(
                zip(
                    range(len(splits)),
                    map(
                        lambda x: BaseTree.get_split_goodness_fit_continuous(*x),
                        zip([arr] * n, [y] * n, splits, [eval_func] * n),
                    ),
                )
            )
        min_eval = min(
            post_split_evals,
            key=lambda x: pipe(
                post_split_evals.get(x), lambda results: results[0] / results[1]
            ),
        )

        return (splits[min_eval], *post_split_evals.get(min_eval))

    def get_optimal_continuous_feature_split(self, X, y, feature_col):
        midpoints = BaseTree.get_valid_midpoints(arr=X[:, feature_col], y=y)
        # midpoints = BaseTree.get_midpoints(sorted_feature)
        if len(midpoints) > 0:
            x = BaseTree.get_min_across_splits_continuous(
                arr=X[:, feature_col], y=y, splits=midpoints, eval_func=self.eval_func
            )

            return x
        else:
            return (0, np.inf, 1)

    @staticmethod
    def get_discrete_split_value(arr, y, eval_func):
        return (
            sum(
                [
                    eval_func(y[arr == value]) * np.sum(arr == value) / len(y)
                    for value in set(arr)
                ]
            ),
            -1
            * sum(
                [
                    pipe(
                        np.sum(arr == value) / len(y),
                        lambda ratio: ratio * np.log(ratio),
                    )
                    for value in set(arr)
                ]
            ),
        )

    def get_optimal_discrete_feature_split(self, X, y, feature_col):
        return BaseTree.get_discrete_split_value(
            X[:, feature_col], y, eval_func=self.eval_func
        )

    def get_terminal_node(self, feature_col, node, feature_value, X, y):
        node_type = self.col_type_map[feature_col]
        if node_type == "continuous":
            if feature_value is None:
                node.children = y
                self.n_nodes += 1
            else:
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
            unique_x_vals = self.discrete_value_maps[feature_col]
            node.update(
                feature_col=feature_col,
                feature_value=None,
                nodes={
                    xval: TreeSplits(children=y[X[:, feature_col] == xval])
                    if np.any(X[:, feature_col] == xval)
                    else TreeSplits(children=y)
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

    def get_discrete_node(self, X, y, feature_col, feature_value, node):
        unique_x_vals = self.discrete_value_maps[feature_col]
        node.update(
            feature_col=feature_col,
            feature_value=feature_value,
            nodes={xval: TreeSplits() for xval in unique_x_vals},
            node_type="discrete",
        )

        for x_col_value in unique_x_vals:
            matches = X[:, feature_col] == x_col_value
            if np.sum(matches) == 0:
                node.nodes[x_col_value] = TreeSplits(
                    node_type="discrete",
                    feature_col=feature_col,
                    feature_value=x_col_value,
                    children=y,
                )
            else:
                self.get_next_split(
                    X=X[matches], y=y[matches], tree_split=node.nodes[x_col_value],
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

        presplit_entropy = self.eval_func(y)

        for k, v in self.col_type_map.items():
            if len(set(X[:, k])) == 1:
                value = np.inf
                split = None
                class_ratios = 1
            elif v == "continuous":

                split, value, class_ratios = self.get_optimal_continuous_feature_split(
                    X=X, y=y, feature_col=k
                )
            else:
                value, class_ratios = self.get_optimal_discrete_feature_split(
                    X=X, y=y, feature_col=k
                )
                split = None

            column_values[k] = (split, value, class_ratios)

        col_idx_with_min_value = max(
            column_values,
            key=lambda x: (presplit_entropy - column_values.get(x)[1])
            / column_values.get(x)[2],
        )

        if (
            self.early_stopping_comparison(
                column_values.get(col_idx_with_min_value)[1], self.early_stopping_value
            )
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
            return self.get_discrete_node(
                X=X,
                y=y,
                feature_value=column_values[col_idx_with_min_value][0],
                feature_col=col_idx_with_min_value,
                node=tree_split,
            )

    def fit(self, X, y):
        self.root = TreeSplits()
        self.discrete_value_maps = {
            col_idx: np.unique(X[:, col_idx])
            for col_idx, col_type in self.col_type_map.items()
            if col_type == "discrete"
        }
        self.get_next_split(X=X, y=y, tree_split=self.root)

    @staticmethod
    def collect_children(node):
        if node.nodes is None or len(node.nodes) == 0:
            return node.children

        return np.concatenate(
            [
                BaseTree.collect_children(child_node)
                for _, child_node in node.nodes.items()
            ]
        ).reshape(-1)

    def predict_from_all_children(self, node):
        children_values = BaseTree.collect_children(node)
        return self.agg_func(children_values)

    def predict_node(self, X, node):
        if node.children is not None and len(node.children):
            return node.children

        if node.node_type == "continuous":
            if X[node.feature_col] > node.feature_value:
                return self.predict_node(X=X, node=node.nodes["above"])
            else:
                return self.predict_node(X=X, node=node.nodes["below"])

        return self.predict_node(X=X, node=node.nodes[X[node.feature_col]])

    def predict(self, X, node=None):
        node = self.root if not node else node
        if X.ndim == 1:
            return self.agg_func(self.predict_node(X=X, node=node))

        return [self.agg_func(self.predict_node(X=row, node=node)) for row in X]


class DecisionTreeClassifier(BaseTree):
    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: Counter(y).most_common(1)[0][0],
            early_stopping_comparison=operator.le,
        )


class DecisionTreeRegressor(BaseTree):
    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: np.mean(y),
            early_stopping_comparison=operator.ge,
        )


class PostPruner:
    def __init__(self, decision_tree, X_validation, y_validation, eval_func):
        self.eval_func = eval_func
        self.tree = decision_tree
        self.X_validation = X_validation
        self.y_validation = y_validation

    def tag_node_from_pruning(self, tree, node, X, y):
        if node.nodes is None or len(node.nodes) == 0:
            return False

        predictions = tree.predict(X)
        whole_tree_score = self.eval_func(y, predictions)

        children = BaseTree.collect_children(node)
        original_nodes = node.nodes
        node.update(
            nodes={},
            children=children,
            feature_col=node.feature_col,
            feature_value=node.feature_value,
            node_type=node.node_type,
        )

        predictions = tree.predict(X)
        pruned_tree_score = self.eval_func(y, predictions)

        if whole_tree_score < pruned_tree_score:
            return True

        node.update(
            children=[],
            nodes=original_nodes,
            feature_col=node.feature_col,
            feature_value=node.feature_value,
            node_type=node.node_type,
        )
        return False

    def prune_node(self, tree, node):
        change_made = self.tag_node_from_pruning(
            tree=tree, node=node, X=self.X_validation, y=self.y_validation
        )
        if not change_made and not node.is_leaf():
            for node_idx, node in node.nodes.items():
                change_made_iter = self.prune_node(tree=tree, node=node)
                change_made = change_made or change_made_iter
            return change_made

        return change_made

    def prune_tree(self):
        tree = copy.deepcopy(self.tree)
        change_made = True
        while change_made:
            change_made = self.prune_node(tree, tree.root)
        return tree
