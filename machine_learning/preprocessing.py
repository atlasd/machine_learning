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
            df_xf = np.where(non_zero, X / self.df_std, X)

        return df_xf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
