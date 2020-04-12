import numpy as np


class Standardizer:
    def __init__(self, mean=True, std=True):
        self.mean = mean
        self.std = std

    def fit(self, X):
        if self.mean:
            self.df_means = X.mean(axis=0)
        if self.std:
            self.df_std = X.std(axis=0)

    def transform(self, X):
        if not self.mean and not self.std:
            return X
        if self.mean:
            df_xf = X - self.df_means
        if self.std:
            non_zero = np.bitwise_not(np.isclose(self.df_std, 0))
            df_xf = np.where(non_zero, df_xf / self.df_std, df_xf)

        return df_xf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MaxScaler:
    """
    Class that scales everything to [-1, 1] interval.
    """

    def fit(self, X):
        # Get the max values
        self.maxes = np.abs(X).max()

    def transform(self, X):
        # Scale by said values
        return X / self.maxes

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
