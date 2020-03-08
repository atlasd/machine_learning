import numpy as np
from toolz import pipe
from collections import Counter
import multiprocessing
import operator
import sys
import copy
from typing import Dict, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sys.setrecursionlimit(10000)


def entropy(y):
    """
    Given a set of class labels, this function
    finds the entropy of the set.
    """
    return -1 * sum(
        [
            pipe(np.sum(y == value) / len(y), lambda ratio: ratio * np.log(ratio))
            for value in set(y)
        ]
    )


def negative_mse(y):
    # Negative mean squared distance from mean for point in Y (variance)
    return -1 * mse(y)


def mse(y):
    # Mean squared distance from mean for point in Y (variance)
    return np.mean((y - np.mean(y)) ** 2)


class TreeSplits:
    """
    This class is used as the node class in a decision tree.
    """

    def __init__(
        self,
        feature_col=None,
        feature_value=None,
        node_type=None,
        nodes=None,
        children=[],
    ):
        self.feature_col = feature_col  # Column idx of split
        self.feature_value = feature_value  # value of split (for continuous)
        self.node_type = node_type  # Continuous or discrete
        self.nodes = nodes  # Children nodes
        self.children = children  # Target values

    def update(self, feature_col, feature_value, node_type, nodes, children=[]):
        """
        Function to update the values of a TreeSplits object.
        """
        self.feature_col = feature_col
        self.feature_value = feature_value
        self.node_type = node_type
        self.nodes = nodes
        self.children = children

    def is_leaf(self):
        """
        Function to determine if a node is a leaf node or not.
        """
        # Has no children nodes
        return self.nodes is None or len(self.nodes) == 0


class BaseTree:
    """
    Base class with decision tree functionality.
    """

    def __init__(
        self,
        col_type_map: Dict,
        eval_func: Callable,
        agg_func: Callable,
        early_stopping_value: float = None,
        early_stopping_comparison: Callable = operator.le,
    ):
        """
        Parameters:
        -----------
        col_type_map : dict
            Mapping from column index to the type ("discrete" or "continuous")
            This dictates the splitting technique used.

        eval_func : Callable
            Takes a set of target values and returns a score. The algorithm
            seeks to maximize the value.

        agg_func : Callable
            The function used for prediction based on the values in a leaf node.

        early_stopping_value : float
            The necessary change in the loss function for the algorithm to continue splitting.
            Defaults to zero if none is passed.

        early_stopping_comparison : Callable
            This determines if value must be less then or greater than value.
            Changes based on regression or classification. Defaults to
            less than or equal to.
        """
        self.agg_func = agg_func
        self.col_type_map = col_type_map
        self.eval_func = eval_func
        # Stops at zero if none is passed
        self.early_stopping_value = (
            0 if not early_stopping_value else early_stopping_value
        )
        self.n_nodes = 1
        self.early_stopping_comparison = early_stopping_comparison

    @staticmethod
    def get_valid_midpoints(arr: np.ndarray, y: np.ndarray):
        """
        Function to get the midpoints between values of `arr` to score
        in determining best split.

        Parameters:
        -----------
        arr : np.ndarray
            The feature array to split on

        y : np.ndarray
            The target array
        """
        # Get sorted indices
        idxs = np.argsort(arr)

        # Get sorted feature array
        sorted_arr = arr[idxs]

        # Get the differences between adjacent values
        arr_diffs = np.diff(sorted_arr)

        # Get the midpoints
        midpoints = sorted_arr[1:] - arr_diffs / 2

        # Get points where differences are greater than 0 (uniqueness) and y values are not the same
        # See report for details on the latter part.
        valid_midpoints = midpoints[np.bitwise_and(arr_diffs > 0, np.diff(y[idxs]) > 0)]

        return valid_midpoints

    @staticmethod
    def get_split_goodness_fit_continuous(
        arr: np.ndarray, y: np.ndarray, split: float, eval_func: Callable
    ):
        """
        Function to evaluate the goodness of the continuous split value.

        Parameters:
        -----------
        arr : np.ndarray
            The feature array to split on

        y : np.ndarray
            The target array

        split : float
            The value to split on

        eval_func : Callable
            The function to evaluate the split on the target
        """
        # Get above and below the split value
        above = arr >= split
        below = arr < split

        # get weighted average eval_func on the splits
        n_above = np.sum(above)
        above_eval = (
            eval_func(y[above]) * n_above / len(y)
        )  # weight = frac points in above
        below_eval = (
            eval_func(y[below]) * (len(y) - n_above) / len(y)
        )  # weight = frac points not in above

        # returns weighted sum of eval_func across splits, and the gain ratio denominator
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
    def get_min_across_splits_continuous(
        arr: np.ndarray, y: np.ndarray, splits: np.ndarray, eval_func: Callable
    ):
        """
        Function to get the best split across many proposed
        splits.


        Parameters:
        -----------
        arr : np.ndarray
            The feature array to split on

        y : np.ndarray
            The target array

        splits : np.ndarray
            The proposed set of split values.

        eval_func : Callable
            The function to evaluate the split on the target
        """
        n = len(splits)
        if n > 500:
            # If many split points, use some threading
            with multiprocessing.Pool(processes=8) as p:
                # Get evaluation scores across all the splits
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
            # If not too many split points, get scores across all splits
            post_split_evals = dict(
                zip(
                    range(len(splits)),
                    map(
                        lambda x: BaseTree.get_split_goodness_fit_continuous(*x),
                        zip([arr] * n, [y] * n, splits, [eval_func] * n),
                    ),
                )
            )
        # Get the minimum split based on gain ratio
        min_eval = min(
            post_split_evals,
            key=lambda x: pipe(
                post_split_evals.get(x),
                lambda results: results[0] / results[1],  # entropy / intrinsic value
            ),
        )

        # Return the best split and the splits scores
        return (splits[min_eval], *post_split_evals.get(min_eval))

    def get_optimal_continuous_feature_split(
        self, X: np.ndarray, y: np.ndarray, feature_col: int
    ):
        """
        Function to get the best continuous split for a column

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix

        y : np.ndarray
            The target vector.

        feature_col : int
            The proposed feature column index
        """
        midpoints = BaseTree.get_valid_midpoints(arr=X[:, feature_col], y=y)
        # If midpoints, get the best one
        if len(midpoints) > 0:
            return BaseTree.get_min_across_splits_continuous(
                arr=X[:, feature_col], y=y, splits=midpoints, eval_func=self.eval_func
            )

        # If no split points, return inf (can't split here)
        return (0, np.inf, 1)

    @staticmethod
    def get_discrete_split_value(arr: np.ndarray, y: np.ndarray, eval_func: Callable):
        """
        Function to get the value of making a discrete split.

        Parameter:
        ----------
        arr : np.ndarray
            The feature array

        y : np.ndarray
            The target array

        eval_func : Callable
            The function to evaluate the splits.
        """

        # First element is the weighted average eval_func of the split
        # Second term is the intrinsic value to penalize many splits.
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

    def get_optimal_discrete_feature_split(
        self, X: np.ndarray, y: np.ndarray, feature_col: int
    ):
        """
        Function to get the best split value for a discrete columns
        """
        return BaseTree.get_discrete_split_value(
            X[:, feature_col], y, eval_func=self.eval_func
        )

    def get_terminal_node(
        self,
        feature_col: int,
        node: TreeSplits,
        feature_value: float,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Function to create a terminal node.

        Parameters:
        -----------
        feature_col : int
            Index of column to use for the split.

        node: TreeSplits
            The node in the tree to create

        feature_value : float
            The value to split on. None if discrete.

        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector
        """
        # Get the node type
        node_type = self.col_type_map[feature_col]

        if node_type == "continuous":
            # If no feature value is passed, this node is the leaf
            if feature_value is None:
                node.children = y
                self.n_nodes += 1
            # If a feature value is passed, create leaves as children
            else:
                # Get the above node
                above = X[:, feature_col] > feature_value

                # Add two children
                node.update(
                    feature_col=feature_col,
                    feature_value=feature_value,
                    node_type=node_type,
                    nodes={
                        "above": TreeSplits(
                            children=y[above]
                        ),  # Children are above points
                        "below": TreeSplits(
                            children=y[np.bitwise_not(above)]
                        ),  # Children are below points
                    },
                )
                # Add two nodes to count
                self.n_nodes += 2
        else:
            # Get the valid values of the discrete column
            unique_x_vals = self.discrete_value_maps[feature_col]
            # Create the node
            node.update(
                feature_col=feature_col,
                feature_value=None,
                nodes={
                    xval: TreeSplits(
                        children=y[X[:, feature_col] == xval]
                    )  # Add in the matching rows
                    if np.any(X[:, feature_col] == xval)  # If discrete values match
                    else TreeSplits(
                        children=y
                    )  # Add in all the rows if there is no values match
                    for xval in unique_x_vals
                },
                node_type="discrete",
            )
            self.n_nodes += len(unique_x_vals)  # increment node counter

    def get_continuous_node(
        self,
        feature_col: int,
        feature_value: float,
        X: np.ndarray,
        y: np.ndarray,
        node: TreeSplits,
    ):
        """
        Function to create a continuous node split.

        Parameters:
        -----------
        feature_col : int
            Index of column to use for the split.

        feature_value : float
            The value to split on. None if discrete.

        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        node: TreeSplits
            The node in the tree to create
        """
        node.update(
            feature_col=feature_col,
            feature_value=feature_value,
            nodes={"below": TreeSplits(), "above": TreeSplits()},
            node_type="continuous",
        )
        # Get the above
        above = X[:, feature_col] >= feature_value
        # Get the next split for the above node
        self.get_next_split(X=X[above], y=y[above], tree_split=node.nodes["above"])
        # Add one node to counter
        self.n_nodes += 1
        # Get the next split for the below node
        self.get_next_split(
            X=X[np.bitwise_not(above)],
            y=y[np.bitwise_not(above)],
            tree_split=node.nodes["below"],
        )
        # Add one node to counter
        self.n_nodes += 1

        return node

    def get_discrete_node(self, X, y, feature_col, feature_value, node):
        """
        Function to create a discrete node split.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        feature_col : int
            Index of column to use for the split.

        feature_value : float
            The value to split on. None if discrete.

        node: TreeSplits
            The node in the tree to create
        """
        # Get the unique values for the X poitns
        unique_x_vals = self.discrete_value_maps[feature_col]

        # Create the node with an empty child for each x value
        node.update(
            feature_col=feature_col,
            feature_value=feature_value,
            nodes={xval: TreeSplits() for xval in unique_x_vals},
            node_type="discrete",
        )

        # For each unique value in the feature column...
        for x_col_value in unique_x_vals:
            # Get the matching rows
            matches = X[:, feature_col] == x_col_value

            # If no matches, put all points in a leaf node
            if np.sum(matches) == 0:
                node.nodes[x_col_value] = TreeSplits(
                    node_type="discrete",
                    feature_col=feature_col,
                    feature_value=x_col_value,
                    children=y,
                )
            else:
                # If there are matches, get the next split
                self.get_next_split(
                    X=X[matches], y=y[matches], tree_split=node.nodes[x_col_value],
                )
                # Increment by one.
                self.n_nodes += 1

        return node

    def get_next_split(self, X: np.ndarray, y: np.ndarray, tree_split: TreeSplits):
        """
        X : np.ndarray
            The feature matrix

        y : np.ndarray
            The target vector

        tree_split : TreeSplit
            The vertex node to use. This allows the tree to track where to
            put the children.
        """
        # If only 1 y value, make a leaf node
        if len(set(y)) == 1:
            tree_split.update(
                feature_col=None,
                feature_value=None,
                node_type=None,
                nodes={},
                children=y,
            )
            return tree_split

        # Get the presplit entropy
        presplit_entropy = self.eval_func(y)

        column_values = {}
        for k, v in self.col_type_map.items():
            # If there's only one value in X, set the split value to infinity
            if len(set(X[:, k])) == 1:
                value = np.inf
                split = None
                class_ratios = 1
            elif v == "continuous":
                # Get the best possible continuous split for the column
                split, value, class_ratios = self.get_optimal_continuous_feature_split(
                    X=X, y=y, feature_col=k
                )
            else:
                # Get the split value for the discrete column
                value, class_ratios = self.get_optimal_discrete_feature_split(
                    X=X, y=y, feature_col=k
                )
                split = None

            column_values[k] = (split, value, class_ratios)

        # Get the column with the largest gain ratio
        col_idx_with_min_value = max(
            column_values,
            key=lambda x: (presplit_entropy - column_values.get(x)[1])
            / column_values.get(x)[2],
        )

        if self.n_nodes < 5:
            logger.info("Column split values:")
            for col, val in column_values.items():
                logger.info(f"Column Index: {col}")
                logger.info(f"Gain Ratio: {(presplit_entropy - val[1]) / val[2]}")

            logger.info(f"Choosing column {col_idx_with_min_value}")

        # If stopping criteria are met or all splits are infinite, terminate the process
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

        # If the best split is continuous, add a continuous node
        if self.col_type_map.get(col_idx_with_min_value) == "continuous":
            return self.get_continuous_node(
                feature_col=col_idx_with_min_value,
                feature_value=column_values[col_idx_with_min_value][0],
                X=X,
                y=y,
                node=tree_split,
            )

        # Otherwise, add a discrete node.
        else:
            return self.get_discrete_node(
                X=X,
                y=y,
                feature_value=column_values[col_idx_with_min_value][0],
                feature_col=col_idx_with_min_value,
                node=tree_split,
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Function to fit the decision tree

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix

        y : np.ndarray
            The target vector
        """
        # Create the root node
        self.root = TreeSplits()

        # Get all possible values for discrete valued columns
        # Necessary so each split can handle unique X values that
        # were not in the training set.
        self.discrete_value_maps = {
            col_idx: np.unique(X[:, col_idx])
            for col_idx, col_type in self.col_type_map.items()
            if col_type == "discrete"
        }

        # Start splitting on the root node.
        self.get_next_split(X=X, y=y, tree_split=self.root)

    @staticmethod
    def collect_children(node: TreeSplits):
        """
        Get all the target values of leaves
        for a subtree. Used for post-pruning

        Parameters:
        -----------
        node : TreeSplits
            The root node of the subtree to collect
        """
        if node.nodes is None or len(node.nodes) == 0:
            return node.children

        # Recursively get all the children and concatenate them
        return np.concatenate(
            [
                BaseTree.collect_children(child_node)
                for _, child_node in node.nodes.items()
            ]
        ).reshape(-1)

    def predict_from_all_children(self, node: TreeSplits):
        """
        Gets the prediction by treating a subtree as a leaf.

        Parameter:
        ----------
        node : TreeSplits
            The root node of the subtree.
        """
        # Collect the children
        children_values = BaseTree.collect_children(node)
        # Aggregate the leaf values
        return self.agg_func(children_values)

    def predict_node(self, X: np.ndarray, node: TreeSplits):
        """
        Make predictions based on a subtree

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix

        node : TreeSplits
            The root node of the subtree.
        """
        # If leaf, return children target values
        if node.children is not None and len(node.children):
            return node.children

        # If continuous, split appropriately, and make recursive call
        if node.node_type == "continuous":
            if X[node.feature_col] > node.feature_value:
                return self.predict_node(X=X, node=node.nodes["above"])
            else:
                return self.predict_node(X=X, node=node.nodes["below"])

        # If discrete, make recusrive call on node.
        return self.predict_node(X=X, node=node.nodes[X[node.feature_col]])

    def predict(self, X: np.ndarray, node: TreeSplits = None):
        """
        Function to make predictions over an X matrix

        Parameters:
        -----------
        X : np.ndarray
             Feature matrix

        node : TreeSplits
            The node to start prediction from.
            If not passed, defaults to the root node.
        """
        node = self.root if not node else node
        if X.ndim == 1:
            # If just one row, predict
            return self.agg_func(self.predict_node(X=X, node=node))

        # If many rows, map prediction over rows.
        return [self.agg_func(self.predict_node(X=row, node=node)) for row in X]


class DecisionTreeClassifier(BaseTree):
    """
    Class using  mode of leaf nodes
    for predictions.

    Works for classification problems
    """

    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: Counter(y).most_common(1)[0][0],  # get mode
            early_stopping_comparison=operator.le,
        )


class DecisionTreeRegressor(BaseTree):
    def __init__(self, col_type_map, eval_func, early_stopping_value=None):
        """
        Class using mean of leaf nodes
        for predictions.

        Works for regression problems
        """
        super().__init__(
            col_type_map=col_type_map,
            eval_func=eval_func,
            early_stopping_value=early_stopping_value,
            agg_func=lambda y: np.mean(y),
            early_stopping_comparison=operator.ge,
        )


class PostPruner:
    """
    Class to run post-pruning with a validation set on
    a BaseTree
    """

    def __init__(
        self,
        decision_tree: BaseTree,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        eval_func: Callable,
    ):
        """
        Parameters:
        decision_tree : BaseTree
            The tree to prune

        X_validation : np.ndarray
            The feature matrix of the validation set

        y_validation : np.ndarray
            The target vector of the validation set

        eval_func : Callable
            The function to evaluate a split
        """
        self.eval_func = eval_func
        self.tree = decision_tree
        self.X_validation = X_validation
        self.y_validation = y_validation

    def tag_node_from_pruning(self, tree, node, X, y):
        """
        Function to test a subtree for pruning

        Parameters:
        -----------
        tree : BaseTree
            The whole tree to predict on

        node : TreeSplits
            The node tagged for pruning

        X : np.ndarray
            The feature matrix for validation

        y : np.ndarray
            The target vector for validation

        """
        # If is a leaf, return False
        if node.nodes is None or len(node.nodes) == 0:
            return False

        # Score predictions from whole tree
        predictions = tree.predict(X)
        whole_tree_score = self.eval_func(y, predictions)

        # Get the children from the node
        children = BaseTree.collect_children(node)
        # Save original nodes
        original_nodes = node.nodes
        # Update node to be a leaf
        node.update(
            nodes={},
            children=children,
            feature_col=node.feature_col,
            feature_value=node.feature_value,
            node_type=node.node_type,
        )

        # Score predictions from leaf
        predictions = tree.predict(X)
        pruned_tree_score = self.eval_func(y, predictions)

        # If leaf is better, don't swap it back and return True for change
        if whole_tree_score < pruned_tree_score:
            return True

        # Otherwise, change the node back to the original node.
        node.update(
            children=[],
            nodes=original_nodes,
            feature_col=node.feature_col,
            feature_value=node.feature_value,
            node_type=node.node_type,
        )
        # Return False (for no change)
        return False

    def prune_node(self, tree: BaseTree, node: TreeSplits):
        """
        Prune a given node

        Parameters:
        -----------
        tree : BaseTree
            The tree to split over

        node : TreeSplits
            The node to tag for pruning
        """
        # Prune node, get if change
        change_made = self.tag_node_from_pruning(
            tree=tree, node=node, X=self.X_validation, y=self.y_validation
        )

        # If change not made and it's not a leaf...
        if not change_made and not node.is_leaf():
            # Prune children nodes
            for node_idx, node in node.nodes.items():
                change_made_iter = self.prune_node(tree=tree, node=node)
                change_made = change_made or change_made_iter  # Track changes
            return change_made

        return change_made

    def prune_tree(self):
        """
        Function to prune a tree.
        """
        tree = copy.deepcopy(self.tree)
        change_made = True
        # As long as changes are made, recursively prune from the root node.
        while change_made:
            change_made = self.prune_node(tree, tree.root)
        return tree
