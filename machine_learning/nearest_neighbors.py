import numpy as np
from toolz import pipe
from machine_learning.validation import metrics
import logging

logger = logging.getLogger(__name__)


class KNearestNeighbors:
    def __init__(self, k, distance_func=lambda x, x0: np.sum(np.subtract(x, x0) ** 2)):
        self.k = k
        self.distance_func = distance_func

    def fit(self, X, y):
        self.X = X
        self.y = y

    def get_distance(self, X, x0):
        return [self.distance_func(row, x0) for row in X]

    def get_nearest_neighbors(self, distances):
        return pipe(
            dict(enumerate(distances)),
            lambda distance_map: sorted(distance_map, key=distance_map.get)[: self.k],
        )

    def predict_prob_single_instance(self, row):
        distances = self.get_distance(X=self.X, x0=row)
        nearest_neighbors = self.get_nearest_neighbors(distances=distances)
        return {
            cls: np.sum(self.y[nearest_neighbors] == cls)
            for cls in np.random.permutation(
                np.unique(self.y)
            )  # Shuffle for coin flip tie breaker
        }

    def predict_prob(self, X):
        return [self.predict_prob_single_instance(row) for row in X]

    def predict(self, X):
        return list(
            map(lambda probs: max(probs, key=probs.get), self.predict_prob(X=X))
        )


class KNearestNeighborRegression(KNearestNeighbors):
    def __init__(self, k, distance_func=lambda x, x0: np.sum(np.subtract(x, x0) ** 2)):
        super().__init__(k=k, distance_func=distance_func)

    def predict_prob_single_instance(self, row):
        distances = self.get_distance(X=self.X, x0=row)
        nearest_neighbors = self.get_nearest_neighbors(distances=distances)
        return np.mean(self.y[nearest_neighbors])

    def predict(self, X):
        return self.predict_prob(X=X)


class EfficientKNNBase:
    def __init__(self, is_edited, knn, k, proportion_cv=0.1, verbose=False):
        self.is_edited = is_edited
        self.knn = knn
        self.k = k
        self.proportion_cv = proportion_cv
        self.verbose = verbose

    def inclusion_of_points(self, X, y, idx, inclusion_exclusion_list):
        if not inclusion_exclusion_list:
            inclusion_exclusion_list.append(idx)
            return True

        closest_element = np.argmin(
            self.knn.get_distance(X=X[inclusion_exclusion_list], x0=X[idx])
        )
        if self.verbose:
            print(f"IDx: {idx}")
            print(f"Row: {X[idx]}")

            print(f"Closest: {X[closest_element]}")
            print(
                f"Class Row: {y[idx]} Class Closest: {y[inclusion_exclusion_list[closest_element]]}"
            )
        if y[inclusion_exclusion_list[closest_element]] != y[idx]:
            if self.verbose:
                print("Adding to Z")
            inclusion_exclusion_list.append(idx)
            return True
        return False

    def exclusion_of_points(self, X, y, idx, inclusion_exclusion_list):
        self.knn.fit(
            np.delete(X, inclusion_exclusion_list + [idx], axis=0),
            np.delete(y, inclusion_exclusion_list + [idx]),
        )
        y_hat = self.knn.predict(X[idx])
        if y_hat[0] != y[idx]:
            inclusion_exclusion_list.append(idx)
            return True
        return False

    def fit(self, X, y):
        inclusion_exclusion = []
        subset_changed = True
        validation_error = [1, 1, 1, 1, 1, 1]
        while subset_changed:
            shuffle_idx = np.random.permutation(np.arange(len(y)))
            n = X.shape[0]
            X_validation = X[shuffle_idx][: int(self.proportion_cv * n)]
            y_validation = y[shuffle_idx][: int(self.proportion_cv * n)]
            subset_changed = False
            print("Looping thru data")

            for idx in shuffle_idx[int(self.proportion_cv * n) :]:
                if idx not in inclusion_exclusion:
                    if self.is_edited:
                        subset_changed_iteration = self.exclusion_of_points(
                            X=X,
                            y=y,
                            idx=idx,
                            inclusion_exclusion_list=inclusion_exclusion,
                        )
                        subset_changed = (
                            True if subset_changed_iteration else subset_changed
                        )
                        self.knn.fit(
                            np.delete(X, inclusion_exclusion, axis=0),
                            np.delete(y, inclusion_exclusion),
                        )
                    else:
                        subset_changed_iteration = self.inclusion_of_points(
                            X=X,
                            y=y,
                            idx=idx,
                            inclusion_exclusion_list=inclusion_exclusion,
                        )
                        subset_changed = (
                            True if subset_changed_iteration else subset_changed
                        )
                        self.knn.fit(X[inclusion_exclusion], y[inclusion_exclusion])

                    # If is edited, allow for early stopping
                    if self.is_edited:
                        predictions = self.knn.predict(X_validation)
                        validation_error.append(
                            1
                            - metrics.accuracy(
                                actuals=y_validation, predictions=predictions
                            )
                        )
                        # If the error is larger than the last 10, early stop
                        if np.argmax(validation_error[-11:]) == 10:
                            break

    def predict_prob(self, X):
        return self.knn.predict_prob(X)

    def predict(self, X):
        return self.knn.predict(X)


class EditedKNN(EfficientKNNBase):
    def __init__(
        self,
        k,
        proportion_cv,
        distance_func=lambda x, x0: np.sum(np.subtract(x, x0) ** 2),
    ):
        super().__init__(
            is_edited=True,
            k=k,
            knn=KNearestNeighbors(k=k, distance_func=distance_func),
            proportion_cv=proportion_cv,
        )


class CondensedKNN(EfficientKNNBase):
    def __init__(
        self,
        k,
        proportion_cv,
        distance_func=lambda x, x0: np.sum(np.subtract(x, x0) ** 2),
    ):
        super().__init__(
            is_edited=False,
            k=k,
            knn=KNearestNeighbors(k=k, distance_func=distance_func),
            proportion_cv=proportion_cv,
        )
