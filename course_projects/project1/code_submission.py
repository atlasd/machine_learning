"""
Course: Introduction to Machine Learning
Assignment: Project 1
Author: David Atlas
"""

import numpy as np
import pandas as pd
from typing import Callable

"""
This section contains some utilities for model building. 
1. Confusion Matrix function
2. KFoldCV - Class for KFold cross validation
3. MulticlassClassifier - Creates one vs. rest classifiers (useful for Winnow2 and boolean outputs
"""


def confusion_matrix(actuals, predictions):
    """Function to find the confusin matrix for a classifier"""

    # Get all the class values
    actual_values = np.sort(np.unique(actuals))

    # Get the dataframe where columns are y and rows are yhat
    return pd.DataFrame(
        {
            y: [
                np.sum((actuals == y) & (predictions == yhat)) for yhat in actual_values
            ]
            for y in actual_values
        },
        index=actual_values,
    )


class KFoldCV:
    """
    Class to handle KFold Cross Validation
    """

    def __init__(self, num_folds: int, shuffle: bool = True):
        """
        Parameters:
        -----------
        num_folds : int
            The number of splits

        shuffle : bool
            If True, rows will be shuffled before the split.
        """
        self.num_folds = num_folds
        self.shuffle = shuffle

    def get_indices(self, X):
        # Get indices of length rows of X. Shuffle if `self.shuffle` is true.
        nrows = X.shape[0]
        return (
            np.random.permutation(
                np.arange(nrows)
            )  # Shuffle the rows if `self.shuffle`
            if self.shuffle
            else np.arange(nrows)
        )

    @staticmethod
    def _get_one_split(split_indices, num_split):
        """
        Given the split indices, get the `num_split` element of the indices.
        """
        return (
            np.delete(
                np.concatenate(split_indices), split_indices[num_split]
            ),  # Drops the test from the train
            split_indices[num_split],  # Gets the train
        )

    @staticmethod
    def _get_indices_split(indices, num_folds):
        # Split the indicies by the number of folds
        return np.array_split(indices, indices_or_sections=num_folds)

    def split(self, X: np.ndarray):
        """
        Creates a generator of train test splits from a matrix X
        """
        # Split the indices into `num_folds` subarray
        indices = self.get_indices(X)
        split_indices = KFoldCV._get_indices_split(
            indices=indices, num_folds=self.num_folds
        )
        for num_split in range(self.num_folds):
            # Return all but one split as train, and one split as test
            yield KFoldCV._get_one_split(split_indices, num_split=num_split)


class MulticlassClassifier:
    """
    Class to do one vs. rest multiclass classification using
    Boolean output classifier.

    """

    def __init__(self, model_cls: Callable, classes: np.ndarray, cls_kwargs):
        """
        Parameters
        ----------
        model_cls : Callable
            A callable that returns the model object to use in fitting.

        classes : np.ndarray
            An array containing the values in `y` for which to create a classifier.

        cls_kwargs : dict
            A dictionary of args for `model_cls` mapping the class value
            to a dictionary of kwargs.
        """
        self.classes = classes
        # Create the models (mapping from class to model)
        self.models = {
            element: model_cls(**cls_kwargs.get(element)) for element in self.classes
        }

    @staticmethod
    def _get_y_binary(y, cls):
        # Transform multivalued outputs into one vs. rest booleans
        # where `cls` is the value of 1.
        return np.where(y == cls, 1, 0)

    def fit(self, X, y):
        """
        Fit the classifiers across all the models.
        """
        if set(y) - set(self.classes):
            raise ValueError("y contains elements not in `classes`")

        for cls, model in self.models.items():
            # Create the binary response for `cls`
            y_binary = MulticlassClassifier._get_y_binary(y, cls)
            # Fit the the model for that class.
            model.fit(X, y_binary)

    def predict(self, X):
        """
        Gets the highest probability class across all the one vs. rest classifiers.
        """
        # Get the prediction_prob across all the classes.
        predictions = {cls: model.predict_prob(X) for cls, model in self.models.items()}

        # Get the class corresponding to the largest probability.
        return [
            max(predictions.keys(), key=lambda x: predictions[x][prediction])
            for prediction in range(X.shape[0])
        ]


"""
This section contains the implementation of our algorithms.
- Winnow2
- Naive Bayes
"""


import numpy as np
from scipy import stats
import pandas as pd
from toolz import pipe
from collections import Counter, OrderedDict


class NaiveBayes:
    def __init__(
        self, column_distribution_map: dict, alpha: float = 1, binomial: bool = False
    ):
        """
        Class to fit Naive Bayes.

        Parameters:
        -----------
        column_distribution_map : dict
            A dictionary that maps each column index to either
            "gaussian" or "multinomial". This will indicate which
            distribution each column should be fitted to.

        alpha : float
            This is the smoothing parameter alpha for multinomial
            distribution.s

        binomial : bool
            If the output is a binomial (boolean), set to true.
            This is only really used to produce proper prediction probabilities
            for the multiclass classification class.
        """
        self.binomial = binomial
        self.column_distribution_map = column_distribution_map
        self.fitted_distributions = {}
        self.is_fitted = False
        self.alpha = alpha

    def _fit_gaussian(self, X, col_idx, y):
        """
        Fits classwise Gaussian distributions to `X[:, col_idx]`
        using the sample parameter MLEs.

        Parameters
        ----------
        X : np.ndarray
            Matrix of features.

        col_idx : int
            The column index for the column to fit the Gaussian to.

        y : np.ndarray
            Vector of target classes
        """
        # Dictionary to map each value in `y` to a Gaussian.
        return {
            val: stats.norm(
                loc=X[y == val, col_idx].mean(),  # Class sample mean
                scale=max(X[y == val, col_idx].std(), 0.00001),  # Class sample std
            )
            for val in sorted(set(y))
        }

    def _fit_multinomial(self, X, col_idx, y):
        """
        Fits classwise multinomial distributions to `X[:, col_idx]`
        using the sample parameter MLEs.

        Parameters
        ----------
        X : np.ndarray
            Matrix of features.

        col_idx : int
            The column index for the column to fit the multinomial to.

        y : np.ndarray
            Vector of target classes
        """
        fitted_distributions = {}
        all_X_values = list(range(int(X[:, col_idx].max()) + 1))
        # For each class...
        for val in sorted(set(y)):
            n = np.sum(y == val)  # Number of instances in the class
            relevant_subset = X[y == val, col_idx]  # Rows in X belonging to class
            value_counts = Counter(
                relevant_subset
            )  # Counts of the values in X in the class
            all_x_value_counts_smoothed = OrderedDict(
                {
                    x_val: self.alpha  # Just alpha if no values
                    if x_val not in value_counts
                    else value_counts[x_val]
                    + self.alpha  # Alpha + Num value occurences otherwise
                    for x_val in all_X_values  # across the values in the column of X
                }
            )
            # n + Alpha * m
            normalizer = n + self.alpha * len(all_X_values)

            # Create the distribution for each class.
            fitted_distributions[val] = stats.multinomial(
                n=n, p=np.array(list(all_x_value_counts_smoothed.values())) / normalizer
            )
        return fitted_distributions

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the classifier across all classes.
        """

        # For each feature column index in X
        for col_idx in range(X.shape[1]):
            if col_idx not in self.column_distribution_map:
                raise ValueError(f"No distribution given for column {col_idx}")

            # If the column has a multinomial tag, fit a multinomial.
            if self.column_distribution_map[col_idx] == "multinomial":
                self.fitted_distributions[col_idx] = self._fit_multinomial(
                    X=X, col_idx=col_idx, y=y
                )
            # Otherwise fit a Gaussian
            elif self.column_distribution_map[col_idx] == "gaussian":
                self.fitted_distributions[col_idx] = self._fit_gaussian(
                    X=X, col_idx=col_idx, y=y
                )

        self.is_fitted = True
        # The prior P(C) gets set to multinomial with p as the
        # proportion of observations in each class C
        self.prior = stats.multinomial(
            n=len(y), p=[np.sum(y == val) / len(y) for val in sorted(set(y))]
        )

    def _predict_one_class(self, X: np.ndarray, class_idx: int):
        """
        Generate prediction value for one class.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix
        class_idx : int
            The index of the class to get prediction value for.

        The output here is the production across features for a given class
        """
        return (
            np.array(
                [
                    self.fitted_distributions[col_idx][class_idx].pdf(X[:, col_idx])
                    if self.column_distribution_map[col_idx] == "gaussian"
                    else self.fitted_distributions[col_idx][class_idx].p[
                        X[:, col_idx].astype("int")
                    ]
                    for col_idx in range(X.shape[1])
                ]
            ).prod(axis=0)
            * self.prior.p[class_idx]
        )

    def predict_prob(self, X):
        """
        Get the prediction probability for each row in X, for each class in y.
        """
        if not self.is_fitted:
            raise ValueError("Must fit model before predictions can be made")

        return pipe(
            [
                self._predict_one_class(
                    X=X, class_idx=class_idx
                )  # Get one class prediction
                for class_idx in self.fitted_distributions[0].keys()  # For each class
            ],
            np.vstack,  # Create a matrix where each row is prob of column being class
            # If self.binomial, return prob of C == 1, else return all rows.
            # Primarily for the multiclass classifier class.
            lambda arr: arr[1] if self.binomial else arr,
        )

    def predict(self, X):
        # Get the class prediction (argmax across classes)
        return np.argmax(self.predict_prob(X), axis=0)
