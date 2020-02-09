import numpy as np
from typing import Callable


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
