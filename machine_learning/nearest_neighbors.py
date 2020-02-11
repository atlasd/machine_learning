import numpy as np


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
        return sorted(enumerate(distances), key=lambda x: x[1])[0][: self.k]

    @staticmethod
    def factorial(n):
        factorial = 1
        for val in range(1, n + 1):
            factorial *= val
        return factorial

    @staticmethod
    def double_factorial(n):
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

    def predict_prob(self, X):
        class_densities = {}
        for row in X:
            distances = self.get_distance(X=self.X, x0=row)
            nearest_neighbors = self.get_nearest_neighbors(distances=distances)
            for cls in set(self.y):
                k_i = np.sum(self.y[nearest_neighbors] == cls)
                N_i = np.sum(self.y == cls)
                r = self.distance_func(nearest_neighbors[-1], row)
                V_k = KNearestNeighbors.get_unit_ball_volume(X.shape[1]) * r
                class_densities[cls] = k_i / (N_i * V_k)
