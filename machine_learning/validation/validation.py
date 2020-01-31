import numpy as np
import pandas as pd


class KFoldCV:
    def __init__(self, num_folds, shuffle=True):
        self.num_folds = num_folds
        self.shuffle = shuffle

    def get_indices(self, X):
        # Get indices of length rows of X. Shuffle if `self.shuffle` is true.
        nrows = X.shape[0]
        return np.random.shuffle(np.arange(nrows)) if self.shuffle else np.arange(nrows)

    @staticmethod
    def _get_one_split(split_indices, num_split):
        return (
            np.delete(np.concatenate(split_indices), split_indices[num_split]),
            split_indices[num_split],
        )

    @staticmethod
    def _get_indices_split(indices, num_folds):
        return np.array_split(indices, indices_or_sections=num_folds)

    def split(self, X):
        # Split the indices into `num_folds` subarray
        indices = self.get_indices(X)
        split_indices = KFoldCV._get_indices_split(
            indices=indices, num_folds=self.num_folds
        )
        for num_split in range(self.num_folds):
            # Return all but one split as train, and one split as test
            yield KFoldCV._get_one_split(split_indices, num_split=num_split)
