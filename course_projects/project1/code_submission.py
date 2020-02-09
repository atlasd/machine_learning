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
