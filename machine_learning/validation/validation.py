import numpy as np
import pandas as pd


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


class KFoldStratifiedCV:
    """
    Class to conduct Stratified KFold CV
    """

    def __init__(self, num_folds, shuffle=True):
        self.num_folds = num_folds
        self.shuffle = shuffle

    def add_split_col(self, arr):
        arr = arr if not self.shuffle else np.random.permutation(arr)
        n = len(arr)
        k = int(np.ceil(n / self.num_folds))
        return pd.DataFrame(
            {"idx": arr, "split": np.tile(np.arange(self.num_folds), k)[0:n],}
        )

    def split(self, y):
        """
        Takes an array of classes, and creates
        train/test splits with proportional examples for each
        group.

        Parameters
        ----------
        y : np.array
            The array of class labels.
        """
        # Make sure y is an array
        y = np.array(y) if isinstance(y, list) else y

        # Groupby y and add integer indices.
        df_with_split = (
            pd.DataFrame({"y": y, "idx": np.arange(len(y))})
            .groupby("y")["idx"]
            .apply(self.add_split_col)  # Add col for split for instance
        )

        # For each fold, get train and test indices (based on col for split)
        for cv_split in np.arange(self.num_folds - 1, 0, -1):
            train_bool = df_with_split["split"] != cv_split
            test_bool = ~train_bool
            # Yield index values of not cv_split and cv_split for train, test
            yield df_with_split["idx"].values[train_bool.values], df_with_split[
                "idx"
            ].values[test_bool.values]
