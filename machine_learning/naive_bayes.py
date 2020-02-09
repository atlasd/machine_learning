import numpy as np
from scipy import stats
import pandas as pd
from toolz import pipe
from collections import Counter, OrderedDict


class NaiveBayes:
    def __init__(self, column_distribution_map, alpha=1, binomial=False):
        self.binomial = binomial
        self.column_distribution_map = column_distribution_map
        self.fitted_distributions = {}
        self.is_fitted = False
        self.alpha = alpha

    def _fit_gaussian(self, X, col_idx, y):
        return {
            val: stats.norm(
                loc=X[y == val, col_idx].mean(),
                scale=max(X[y == val, col_idx].std(), 0.00001),
            )
            for val in sorted(set(y))
        }

    def _fit_multinomial(self, X, col_idx, y):
        fitted_distributions = {}
        all_X_values = list(range(X[:, col_idx].max() + 1))
        for val in sorted(set(y)):
            n = np.sum(y == val)
            relevant_subset = X[y == val, col_idx]
            value_counts = Counter(relevant_subset)
            all_x_value_counts_smoothed = OrderedDict(
                {
                    x_val: self.alpha
                    if x_val not in value_counts
                    else value_counts[x_val] + self.alpha
                    for x_val in all_X_values
                }
            )
            normalizer = n + self.alpha * len(all_X_values)
            fitted_distributions[val] = stats.multinomial(
                n=n, p=np.array(list(all_x_value_counts_smoothed.values())) / normalizer
            )
        return fitted_distributions

    def fit(self, X, y):
        for col_idx in range(X.shape[1]):
            if col_idx not in self.column_distribution_map:
                raise ValueError(f"No distribution given for column {col_idx}")

            if self.column_distribution_map[col_idx] == "multinomial":
                self.fitted_distributions[col_idx] = self._fit_multinomial(
                    X=X, col_idx=col_idx, y=y
                )

            elif self.column_distribution_map[col_idx] == "gaussian":
                self.fitted_distributions[col_idx] = self._fit_gaussian(
                    X=X, col_idx=col_idx, y=y
                )

        self.is_fitted = True
        self.prior = stats.multinomial(
            n=len(y), p=[np.sum(y == val) / len(y) for val in sorted(set(y))]
        )

    def _predict_one_class(self, X, class_idx):
        return np.array(
            [
                self.fitted_distributions[col_idx][class_idx].pdf(X[:, col_idx])
                if self.column_distribution_map[col_idx] == "gaussian"
                else self.fitted_distributions[col_idx][class_idx].p[
                    X[:, col_idx].astype("int")
                ]
                for col_idx in range(X.shape[1])
            ]
        ).prod(axis=0)

    def predict_prob(self, X):
        if not self.is_fitted:
            raise ValueError("Must fit model before predictions can be made")

        return pipe(
            [
                self._predict_one_class(X=X, class_idx=class_idx)
                for class_idx in self.fitted_distributions[0].keys()
            ],
            np.vstack,
            lambda arr: arr[1] if self.binomial else arr,
        )

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=0)
