import numpy as np
import pandas as pd
from itertools import product
from typing import Callable, Dict, Union


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
            np.setdiff1d(np.concatenate(split_indices), split_indices[num_split]),
            split_indices[num_split],
        )

    @staticmethod
    def _get_indices_split(indices, num_folds):
        # Split the indicies by the number of folds
        return np.array_split(indices, indices_or_sections=num_folds)

    def split(self, X: np.ndarray, y: np.ndarray = None):
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

    def split(self, y, X=None):
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
        for cv_split in np.arange(self.num_folds - 1, -1, -1):
            train_bool = df_with_split["split"] != cv_split
            test_bool = ~train_bool
            # Yield index values of not cv_split and cv_split for train, test
            yield df_with_split["idx"].values[train_bool.values], df_with_split[
                "idx"
            ].values[test_bool.values]


class GridSearchCV:
    """
    Class to assist with grid searching over potential parameter values.
    """

    def __init__(
        self,
        model_callable: Callable,
        param_grid: Dict,
        scoring_func: Callable,
        cv_object: Union[KFoldCV, KFoldStratifiedCV] = None,
        X_validation=None,
        y_validation=None,
    ):
        """
        Parameters:
        -----------
        model_callable : Callable
            Function that generates a model object. Should
            take the keys of param_grid as arguments.

        param_grid : dict
            Mapping of arguments to potential values

        scoring_func : Callable
            Takes in y and yhat and returns a score to be maximized.

        cv_object
            A CV object from above that will be used to make validation
            splits.

        X_validation: np.ndarrary
            X validation set. If not passed, CV is used.

        y_validation: np.ndarrary
            y validation set. If not passed, CV is used.


        """
        self.model_callable = model_callable
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_object = cv_object
        self.X_val = X_validation
        self.y_val = y_validation

    @staticmethod
    def create_param_grid(param_grid: Dict):
        """
        A mapping of arguments to values to grid search over.

        Parameters:
        -----------
        param_grid : Dict
            {kwarg: [values]}
        """
        return (
            dict(zip(param_grid.keys(), instance))
            for instance in product(*param_grid.values())
        )

    def get_single_fitting_iteration(self, model, X: np.ndarray, y: np.ndarray):
        """
        Run a model fit and validate step.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for training.

        y : np.ndarray
            Target vector for training

        model
            Model object with a fit and predict method.
        """
        scores = []

        if self.cv_object:
            # Create train/test splits
            for train, test in self.cv_object.split(X=X, y=y):
                # Fit the model
                model.fit(X[train], y[train])
                # Get the predictions
                yhat = model.predict(X[test])
                # Get the scores
                scores.append(self.scoring_func(y[test], yhat))
        else:
            model.fit(X, y)
            yhat = model.predict(self.X_val)
            scores.append(self.scoring_func(self.y_val, yhat))

        # Get the average score.
        return np.mean(scores)

    def get_cv_scores(self, X: np.ndarray, y: np.ndarray):
        """
        Runs the grid search across the parameter grid.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector
        """
        # Create the parameter grid
        param_grid = list(GridSearchCV.create_param_grid(self.param_grid))

        # Zip the grid to the results from a single fit
        return zip(
            param_grid,
            [
                self.get_single_fitting_iteration(
                    X=X, y=y, model=self.model_callable(**param_set)
                )
                for param_set in param_grid
            ],
        )
