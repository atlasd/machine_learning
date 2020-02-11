import numpy as np
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class Winnow2:
    def __init__(
        self,
        weight_scaler: float,
        threshold: float,
        num_features: int,
        verbose: bool = False,
    ):
        self.weight_scaler = weight_scaler
        self.threshold = threshold
        self.num_features = num_features
        self.weights = np.ones((num_features,))
        self.verbose = verbose

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
        """
        Function to either promote or demote, based on whether division or
        multiplication is passed as the scaling function.
        """
        if isinstance(X, list):
            X = np.array(X)

        if self.verbose:
            logger.info(f"Initial weights: {self.weights}")
            logger.info(f"Training instance: {X}")

        new_weights = np.where(
            X == 1, scale_func(self.weights, self.weight_scaler), self.weights
        )

        if self.verbose:
            logger.info(f"Updated weights: {new_weights}")

        return new_weights

    def promote_weights(self, X: np.array):
        if self.verbose:
            logger.info("Promoting weights...")
        return self.adjust_weights(X=X, scale_func=np.multiply)

    def demote_weights(self, X: np.array):
        if self.verbose:
            logger.info("Demoting weights...")
        return self.adjust_weights(X=X, scale_func=np.true_divide)

    def run_training_iteration(self, X: np.array, y: bool):
        """
        Runs a single training iteration for Winnow2.
        """
        yhat = self.predict(X)
        if self.verbose:
            logger.info(f"Actual: {y} Prediction: {yhat}")

        # If prediction is correct, do nothing
        if yhat == y:
            if self.verbose:
                logger.info("Correct prediction. No updates.")
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
