import numpy as np
from toolz import pipe


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

    @staticmethod
    def factorial(n):
        if n % 1 > 0:
            raise ValueError("n must be integer")
        n = int(n)
        factorial = 1
        for val in range(1, n + 1):
            factorial *= val
        return factorial

    @staticmethod
    def double_factorial(n):
        if n % 1 > 0:
            raise ValueError("n must be integer")
        n = int(n)
        factorial = 1
        for val in range(1, n + 1):
            if n % 2 == val % 2:
                factorial *= val
        return factorial

    @staticmethod
    def get_unit_ball_volume(n):
        if n % 2 == 0:
            return (np.pi ** (n / 2)) / KNearestNeighbors.factorial(n / 2)
        return (
            (np.pi ** np.floor(n / 2)) * (2 ** np.ceil(n / 2))
        ) / KNearestNeighbors.double_factorial(n)

    def predict_prob_single_instance(self, row):
        distances = self.get_distance(X=self.X, x0=row)
        nearest_neighbors = self.get_nearest_neighbors(distances=distances)
        return {
            cls: np.sum(self.y[nearest_neighbors] == cls)
            for cls in np.random.permutation(
                np.unique(self.y)
            )  # Shuffle for coin flip tie breaker
        }

    def predict_prob(self, X_q):
        return [self.predict_prob_single_instance(row) for row in X_q]

    def predict(self, X_q):
        return list(
            map(lambda probs: max(probs, key=probs.get), self.predict_prob(X_q=X_q))
        )


class EditedKNN:
    def __init__(self, knn, k=1):
        self.knn = knn
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.knn.fit(X, y)
        y_hat = self.knn.predict(X, y)
        self.keep_idx = np.arange(len(y))[y_hat == y]
        self.knn.fit(self.X[self.keep_idx], self.y[self.keep_idx])

    def predict_prob(self, X):
        return self.knn.predict_prob(X)

    def predict(self, X):
        return self.knn.predict(X)


class CondensedKNN:
    def __init__(self, knn, k=1):
        self.knn = knn
        self.k = k
