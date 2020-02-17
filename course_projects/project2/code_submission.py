import numpy as np
from toolz import pipe
import logging
from typing import Callable, List

logger = logging.getLogger(__name__)


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
