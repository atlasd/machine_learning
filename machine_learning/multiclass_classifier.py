import numpy as np
from typing import Callable


class MulticlassClassifier:
    def __init__(self, model_cls: Callable, classes: np.ndarray, cls_kwargs):
        self.classes = classes
        self.models = {
            element: model_cls(**cls_kwargs.get(element)) for element in self.classes
        }

    @staticmethod
    def _get_y_binary(y, cls):
        return np.where(y == cls, 1, 0)

    def fit(self, X, y):
        if set(y) - set(self.classes):
            raise ValueError("y contains elements not in `classes`")

        for cls, model in self.models.items():
            y_binary = MulticlassClassifier._get_y_binary(y, cls)
            model.fit(X, y_binary)

    def predict(self, X):
        predictions = {cls: model.predict_prob(X) for cls, model in self.models.items()}
        return [
            max(predictions.keys(), key=lambda x: predictions[x][prediction])
            for prediction in range(X.shape[0])
        ]
