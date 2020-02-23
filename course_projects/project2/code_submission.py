from itertools import product
from typing import Callable, Dict, Union, List
import logging
import pandas as pd
import warnings
import numpy as np
from toolz import pipe
import io
import requests as r

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
In the section below, we define our validation and grid search objects.
"""


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
        cv_object: Union[KFoldCV, KFoldStratifiedCV],
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
        """
        self.model_callable = model_callable
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_object = cv_object

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

    def get_single_fitting_iteration(self, X: np.ndarray, y: np.ndarray, model):
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
        # Create train/test splits
        for train, test in self.cv_object.split(X=X, y=y):
            # Fit the model
            model.fit(X[train], y[train])
            # Get the predictions
            yhat = model.predict(X[test])
            # Get the scores
            scores.append(self.scoring_func(y[test], yhat))
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
                    X, y, model=self.model_callable(**param_set)
                )
                for param_set in param_grid
            ],
        )


"""
In this section, we define  our NearestNeighbor objects
"""


class KNearestNeighbors:
    """
    Class to perform KNN algorithm for classification.
    """

    def __init__(
        self,
        k: int,
        distance_func: Callable = lambda x, x0: np.sum(np.subtract(x, x0) ** 2),
    ):
        """
        Parameters
        -----------
        k : int
            The number of neighbors to take into account in the voting scheme.

        distance_func : Callable
            Function to calculate distance between two points
            in feature space. Defaults to 2 norm.
        """
        self.k = k
        self.distance_func = distance_func

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Function to 'fit' the  KNN. Doesn't really do anything
        but save the values of `X` and `y`.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix.

        y : np.ndarray
            The target vector
        """
        self.X = X
        self.y = y

    def get_distance(self, X: np.ndarray, x0: np.ndarray):
        """
        Function to get the distances between
        the rows of X and x0

        Parameters
        ----------
        X : np.ndarray
            The matrix of observations

        x0 : np.ndarray
            The row to get the distances against the matrix.
        """
        # Apply distance_func across rows of X
        return [self.distance_func(row, x0) for row in X]

    def get_nearest_neighbors(self, distances: List[float]):
        """
        Get the nearest neighbors given the distances.

        Parameters
        ----------
        distances : List[float]
            The list of distances.
        """
        return pipe(
            dict(enumerate(distances)),  # Map index to distance
            # Sort the indices based on their value in the mapping, and take the first k
            lambda distance_map: sorted(distance_map, key=distance_map.get)[: self.k],
        )

    def predict_prob_single_instance(self, row: np.ndarray):
        """
        Gets the count  within the k neighbors for each class

        Parameters
        ----------
        row : np.ndarray
            The row to do prediction for
        """
        # Get the pairwise distances with X
        distances = self.get_distance(X=self.X, x0=row)

        # Get the k nearest neighbors
        nearest_neighbors = self.get_nearest_neighbors(distances=distances)

        # For each class, get the number of the neighbors that is in that class
        return {
            cls: np.sum(self.y[nearest_neighbors] == cls)
            for cls in np.random.permutation(
                np.unique(self.y)
            )  # Shuffle for coin flip tie breaker (sorted takes the first in ties)
        }

    def predict_prob(self, X: np.ndarray):
        """
        Apply the above across 1 or many instances
        """
        if X.ndim == 1:
            return [self.predict_prob_single_instance(X)]
        return [self.predict_prob_single_instance(row) for row in X]

    def predict(self, X: np.ndarray):
        """
        Make predictions for the rows in X

        Parameters:
        ---------
        X : np.ndarray
            The matrix to make predictions for.
        """
        # Get the largest count across all the classes for each row in X
        return list(
            map(lambda probs: max(probs, key=probs.get), self.predict_prob(X=X))
        )


class KNearestNeighborRegression(KNearestNeighbors):
    """
    Class for KNN for Regression problems
    """

    def __init__(
        self,
        k: int,
        distance_func: Callable = lambda x, x0: np.sum(np.subtract(x, x0) ** 2),
    ):
        """
        Parameters
        -----------
        k : int
            The number of neighbors to take into account in the voting scheme.

        distance_func : Callable
            Function to calculate distance between two points
            in feature space. Defaults to 2 norm.
        """
        super().__init__(k=k, distance_func=distance_func)

    def predict_prob_single_instance(self, row: np.ndarray):
        """
        Gets the mean target value within the k neighbors

        Parameters
        ----------
        row : np.ndarray
            The row to do prediction for
        """
        # Get distances
        distances = self.get_distance(X=self.X, x0=row)
        # Get neighbors
        nearest_neighbors = self.get_nearest_neighbors(distances=distances)
        # Get mean of y of neighbors
        return np.mean(self.y[nearest_neighbors])

    def predict(self, X: np.ndarray):
        """
        Make predictions for the rows in X

        Parameters:
        -----------
        X : np.ndarray
            The matrix to make predictions for
        """
        return self.predict_prob(X=X)


class CondensedKNN:
    """
    Class to apply the Condensed KNN algorithm.
    """

    def __init__(
        self, verbose: bool = False, knn: KNearestNeighbors = KNearestNeighbors(k=1)
    ):
        """
        Parameters:
        -----------
        verbose : bool
            If true, will print intermediate results

        knn : KNearestNeighbors
            The underlying KNN object that will be used for making predictions.

        """
        self.knn = knn
        self.knn.k = 1  # Make sure k = 1 always
        self.verbose = verbose

    def get_inclusion(self, X: np.ndarray, y: np.ndarray, idx: int, Z: List[int]):
        """
        Function to run the step to see if row `idx` should be
        included in the condensed set.

        Parameters:
        ----------
        X : np.ndarray
            The feature matrix for training

        y : np.ndarray
            The labels for training

        idx : int
            The proposed row of X to include in Z

        Z : List[int]
            THe list of rows that have already been included
        """
        # Fit the KNN on just the rows of Z. Remember that k = 1.
        self.knn.fit(X[Z], y[Z])

        # If the prediction is incorrect
        if self.knn.predict(X[idx]) != y[idx]:
            # Add idx to Z
            Z.append(idx)

            # Return True (change to Z has occured)
            return True

        return False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Function to run the condensed KNN fitting procedure.

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector
        """
        # Track changes to Z
        is_changed = True

        # Z begins as empty set
        Z = []
        while is_changed:
            is_changed = False
            # If Z is empty, add the first element of X
            if not Z:
                Z.append(0)

            # Iterate through rows of X
            for idx in range(1, len(X)):
                # If the row is not already in Z
                if idx not in Z:
                    # Run inclusion procedure
                    changed = self.get_inclusion(X, y, idx, Z)
                    # Track changes to Z
                    is_changed = True if changed else is_changed
        # Fit model over rows of Z
        self.knn.fit(X[Z], y[Z])

    def predict(self, X: np.ndarray):
        """
        Function to predict for rows of X

        Parameters:
        ---------
        X : np.ndarray
            The matrix to make predictions for.
        """
        return self.knn.predict(X)


class EditedKNN:
    """
    Performs the EditedKNN algorithm for classification.
    """

    def __init__(self, k, proportion_cv=0.1, verbose=False):
        """
        Parameters
        ----------
        k : int
            The number of neighbors to consider in the voting scheme

        proportion_cv : float
            The proportion of the training set that should be used
            for validation (on when to stop training).

        verbose : bool
            If True, will log intermediate steps.

        """
        self.knn = KNearestNeighbors(k=k)
        self.k = k
        self.proportion_cv = proportion_cv
        self.verbose = verbose

    def get_exclusion(self, X, y, idx, Z):
        """
        Function to determine if the `idx` row of X should be
        excluded from the final training set


        Parameters
        ----------
        X : np.ndarray
            The feature matrix.

        y : np.ndarray
            The target vector

        idx : int
            The row of X under consideration

        Z : List[int]
            The list of indices that are currently included.
        """

        # Remove idx from Z
        Z.remove(idx)
        # Fit the model
        self.knn.fit(X[Z], y[Z])
        # If classification is incorrect, put idx back in Z
        if y[idx] != self.knn.predict(X[idx]):
            Z.append(idx)
            return False
        return True

    def validation_error_decreasing(self, X_val, y_val, last_validation_score):
        """
        Function to get validation scores

        Parameters
        ----------
        X_val : np.ndarray
            The feature matrix for the validation set

        y_val : np.ndarray
            The target vector for the validation set

        last_validation_score : float
            The previous validation score
        """
        error = np.mean(np.array(self.knn.predict(X_val)) != np.array(y_val))
        return error < last_validation_score, error

    def fit(self, X, y):
        """
        Function to run the edited fitting procedure.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix.

        y : np.ndarray
            The target vector
        """
        # Split off subset for validation
        n_holdout = int(len(X) * self.proportion_cv)
        X_validate = X[:n_holdout]
        y_validate = y[:n_holdout]
        X_train = X[n_holdout:]
        y_train = y[n_holdout:]

        # Starting edited set with all indices in it.
        Z = list(range(len(X_train)))

        # tracking validation scores
        validation_decreasing = True
        last_validation_score = np.inf

        # tracking changes to the edited set
        is_changed = True

        # While changes to edit set and validation scores decreasing...
        while is_changed and validation_decreasing:
            is_changed = False

            # For each row in X
            for idx in range(len(X_train)):
                # Only indices that are still in Z can be eliminated
                if idx in Z:
                    # Run the exclusion
                    changed = self.get_exclusion(X_train, y_train, idx, Z)
                    # Track if changes are made
                    is_changed = True if changed else is_changed

            # Fit the model on the edited set and get validation scores
            self.knn.fit(X[Z], y[Z])
            (
                validation_decreasing,
                last_validation_score,
            ) = self.validation_error_decreasing(
                X_val=X_validate,
                y_val=y_validate,
                last_validation_score=last_validation_score,
            )

    def predict(self, X):
        """
        Make predictions for the rows in X

        Parameters:
        ---------
        X : np.ndarray
            The matrix to make predictions for.
        """
        return self.knn.predict(X)


"""
In this section, we apply the algorithms to the two classification problems. We define some 
helper functions.
"""


class Standardizer:
    """
    Class to standardize input features.
    """

    def __init__(self, mean=True, std=True):
        self.mean = mean
        self.std = std

    def fit(self, X):
        """
        Calculates the columnwise mean and standard deviation
        """
        if self.mean:
            self.df_means = X.mean(axis=0)  # Get the colwise means
        if self.std:
            self.df_std = X.std(axis=0)  # Get the colwise stds

    def transform(self, X):
        """
        Applies the columnwise mean and std transformations
        """
        if not self.mean and not self.std:
            return X
        if self.mean:
            df_xf = X - self.df_means  # Subtract means
        if self.std:
            is_zero = np.isclose(self.df_std, 0)  # If non-zero variance,
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_xf = np.where(
                    is_zero, X, X / self.df_std
                )  # Ensure no divide by zero issues

        return df_xf

    def fit_transform(self, X):
        """
        Fits on the columns and then transforms X
        """
        self.fit(X)
        return self.transform(X)


def accuracy(actuals, predictions):
    """
    Function to get classifier accuracy
    """
    return np.mean(actuals == predictions)


def mean_squared_error(actuals, predictions):
    """
    Function to get MSE
    """
    return np.mean((actuals - predictions) ** 2)


def choose_k(
    X,
    y,
    model_call,
    param_grid,
    scoring_func=accuracy,
    cv=KFoldStratifiedCV(num_folds=3),
):
    """
    Function to use cross-validation to choose a value of k

    Parameters:
    -----------
    X : np.ndarray
        The feature matrix

    y : np.ndarray
        The target vector

    model_call : Callable
        A function that returns a model object. Its arguments must be the
        keys in param_grid.

    param_grid : Dict
        Mapping of arguments to values to try

    scoring_func : Callable
        The function that scores the results of a model.
        This value is maximized.

    cv
        The validation object to use for the cross validation.
    """
    grid_search_cv = GridSearchCV(
        model_callable=model_call,
        param_grid=param_grid,
        scoring_func=scoring_func,
        cv_object=cv,
    )
    # Get the last sorted value and take k from that values
    return sorted(list(grid_search_cv.get_cv_scores(X, y)), key=lambda x: x[1])[-1][0][
        "k"
    ]


def run_experiment(
    X,
    y,
    model_call,
    param_grid=None,
    scoring_func=accuracy,
    cv=KFoldStratifiedCV(num_folds=5),
):
    """
    Runs a single experiment. If a param_grid
    is passed, it will select `k` from the values passed.
    """
    scores = []
    iteration = 0
    # Iterate through the split
    for train, test in cv.split(y):
        # If first iteration and k values are passed, get the best one
        if iteration == 0 and param_grid:
            k = choose_k(
                X[train], y[train], model_call, param_grid, scoring_func, cv=cv
            )
            logger.info(f"Choosing k={k}")
        else:
            # Defaults to 1 for condensed.
            k = 1

        iteration += 1

        # Instantiate the model with the value of k
        model = model_call(k=k)

        # Standardize the data
        standardizer = Standardizer(mean=True, std=True)

        # Fit the model
        model.fit(X=standardizer.fit_transform(X[train]), y=y[train])

        # make test set predictions
        y_pred = model.predict(X=standardizer.transform(X[test]))

        # Append the score
        scores.append(scoring_func(y[test], y_pred))
    logger.info(f"Avg Score: {np.mean(scores)}")
    return model


"""
In this section, we load and clean the data, and run the experiments
"""


if __name__ != "__main__":
    logger.info("Running Ecoli Experiment")
    np.random.seed(73)
    df = pd.read_csv(
        io.StringIO(
            r.get(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
            )
            .text.replace("   ", " ")
            .replace("  ", " ")
        ),
        sep=" ",
        header=None,
        names=[
            "id",
            "mcg",
            "gvh",
            "lip",
            "chg",
            "aac",
            "alm1",
            "alm2",
            "instance_class",
        ],
    )

    data = df.drop("id", axis=1).sample(frac=1)

    y = data["instance_class"].astype("category").cat.codes.values
    X = data.drop(axis=1, labels="instance_class").values

    logger.info("Running Standard KNN")
    run_experiment(
        X,
        y,
        model_call=lambda k: KNearestNeighbors(k=k),
        param_grid={"k": [1, 2, 3, 4, 5]},
    )

    logger.info("Running Edited KNN")
    run_experiment(
        X,
        y,
        model_call=lambda k: EditedKNN(k=k, proportion_cv=0.1),
        param_grid={"k": [1, 2, 3, 4, 5]},
    )

    logger.info("Running Condensed KNN")
    model = run_experiment(X, y, model_call=lambda k: CondensedKNN(verbose=True))

    logger.info("Running Image Segmentation Experiments")
    image_segmentation = pipe(
        r.get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
        ).text.split("\n"),
        lambda lines: pd.read_csv(
            io.StringIO("\n".join(lines[5:])), header=None, names=lines[3].split(",")
        ),
        lambda df: df.assign(
            instance_class=lambda df: df.index.to_series().astype("category").cat.codes
        ),
    )

    X = image_segmentation.drop(["instance_class", "REGION-PIXEL-COUNT"], axis=1).values
    y = image_segmentation["instance_class"].values

    np.random.seed(73)
    logger.info("Running Standard KNN")
    run_experiment(
        X,
        y,
        model_call=lambda k: KNearestNeighbors(k=k),
        param_grid={"k": [1, 2, 3, 4, 5]},
    )

    logger.info("Running Edited KNN")
    run_experiment(
        X,
        y,
        model_call=lambda k: EditedKNN(k=k, proportion_cv=0.2),
        param_grid={"k": [1, 2, 3, 4, 5]},
    )

    logger.info("Running Condensed KNN")
    run_experiment(X, y, model_call=lambda k: CondensedKNN(verbose=True))

    logger.info("Running CPU Performance Experiment")

    cpu_performance = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        header=None,
        names=[
            "vendor_name",
            "model_name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",
            "ERP",
        ],
    )

    X = cpu_performance.drop(["vendor_name", "model_name", "PRP", "ERP"], axis=1).values
    y_real = cpu_performance["PRP"].values
    y_ols = cpu_performance["ERP"].values

    np.random.seed(73)
    run_experiment(
        X=X,
        y=y_real,
        model_call=lambda k: KNearestNeighborRegression(k=k),
        param_grid={"k": list(range(1, 5))},
        scoring_func=lambda *args, **kwargs: -1
        * np.sqrt(mean_squared_error(*args, **kwargs)),
        cv=KFoldCV(num_folds=5),
    )

    logger.info("Running Forest Fires Regression")

if __name__ == "__main__":
    fires_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    )

    X = (
        fires_data.drop("area", axis=1)
        .pipe(lambda df: pd.get_dummies(df, columns=["month", "day"], drop_first=True))
        .values
    )
    y = fires_data["area"].values

    np.random.seed(73)
    model = run_experiment(
        X=X,
        y=y,
        model_call=lambda k: KNearestNeighborRegression(k=k),
        param_grid={"k": list(range(1, 5))},
        scoring_func=lambda *args, **kwargs: -1
        * np.sqrt(mean_squared_error(*args, **kwargs)),
        cv=KFoldCV(num_folds=5),
    )
