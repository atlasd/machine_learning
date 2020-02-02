import numpy as np
from typing import Callable


class Winnow2:
    def __init__(self, weight_scaler: float, threshold: float, num_features: int):
        self.weight_scaler = weight_scaler
        self.threshold = threshold
        self.num_features = num_features
        self.weights = np.ones((num_features,))

    def predict_prob(self, X: np.array):
        """
        Function to get the raw prediction score (not binary)
        """
        return X @ self.weights

    def predict(self, X: np.array):
        """
        Function to get the binary prediction value.
        """
        return self.predict_prob(X) > self.threshold

    def adjust_weights(self, X: np.array, scale_func: Callable):
        if isinstance(X, list):
            X = np.array(X)
        return np.where(
            X == 1, scale_func(self.weights, self.weight_scaler), self.weights
        )

    def promote_weights(self, X: np.array):
        return self.adjust_weights(X=X, scale_func=np.multiply)

    def demote_weights(self, X: np.array):
        return self.adjust_weights(X=X, scale_func=np.true_divide)

    def run_training_iteration(self, X: np.array, y: bool):
        yhat = self.predict(X)
        # If prediction is correct, do nothing
        if yhat == y:
            return

        # If prediction is 0 and y is 1, promote
        if not yhat and y:
            self.weights = self.promote_weights(X)
            return

        # If prediction is 1 and y is 0, demote
        self.weights = self.demote_weights(X)
        return

    def fit(self, X, y):
        for X_instance, y_instance in zip(X, y):
            self.run_training_iteration(X_instance, y_instance)
