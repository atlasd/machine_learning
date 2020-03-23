import numpy as np
import pandas as pd


class LogisticRegressionClassifier:
    """
    Class to fit a logistic regression classifier
    """

    def __init__(
        self,
        convergence_tolerance=0.1,
        learning_rate=0.01,
        fit_intercept=True,
        max_iter=5,
    ):
        """
        Parameters:
        ----------
        convergence_tolerance : float
            The stopping criteria based on
            the value of the gradient.

        learning_rate : float
            Eta for the gradient update

        fit_intercept : bool
            If true, will add column of ones

        max_iter : int
            Stopping criteria based on the number of updates.

        """
        self.fit_intercept = fit_intercept
        self.convergence_tolerance = convergence_tolerance
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    @staticmethod
    def get_matrix_product(X, weights):
        # Weights by X
        return weights @ X.T

    @staticmethod
    def get_class_scores(X, weights):
        """
        Get the class scores for each of the classes

        Parameters:
        -----------
        X : np.ndarray
            The input matrix
        weights : np.ndarray
            Weights of linear transformation
        """
        matrix_product = LogisticRegressionClassifier.get_matrix_product(
            X=X, weights=weights
        )

        # Get the normalized likelihood for all the classes
        return (np.exp(matrix_product) / np.sum(np.exp(matrix_product), axis=0)).T

    @staticmethod
    def get_gradient_update_term(X, y, weights):
        """
        Get the gradient update term.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        weights : np.ndarray
            Weights of Logistic Regression

        """
        class_scores = LogisticRegressionClassifier.get_class_scores(
            X=X, weights=weights
        )

        # Get the one-hots for y classes
        y_one_hot = pd.get_dummies(y).values

        # Get the gradient (r - y) X
        return np.dot((class_scores - y_one_hot).T, X)

    def fit(self, X, y):
        """
        Run the fitting procedure for the LogisticRegression

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        weights : np.ndarray
            Weights of Logistic Regression

        """
        # Add column of ones if fit_intercept
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Get the classes
        classes = set(y)

        # Initialize random weights around 0
        self.weights = np.random.uniform(
            low=-0.01, high=0.01, size=(len(classes), X.shape[1])
        )

        # Calculate the gradient
        gradient = LogisticRegressionClassifier.get_gradient_update_term(
            X=X, y=y, weights=self.weights
        )

        iter_count = 1

        # While convergence criteria not met
        while (
            np.any(np.abs(gradient) > self.convergence_tolerance)
            and iter_count < self.max_iter
        ):
            # Update weights
            self.weights = self.weights - self.learning_rate * gradient

            # Calculate weights
            gradient = LogisticRegressionClassifier.get_gradient_update_term(
                X=X, y=y, weights=self.weights
            )

            # Increment count
            iter_count += 1

            # Stop if gradient explodes
            if pd.isnull(gradient).any():
                print("Exploding gradient")
                break

    def predict_probs(self, X):
        """
        Get output scores for predictions on X

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix
        """

        # Add ones if fit_intercept
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        # Make predictions
        return self.weights @ X.T

    def predict(self, X):
        """
        Make predictions on X

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix
        """

        # Get the largest class prediction
        return np.argmax(self.predict_probs(X), axis=0)


class AdalineNetwork:
    """
    Class to fit Adaline Network
    """

    def __init__(
        self,
        convergence_tolerance=0.1,
        learning_rate=0.01,
        fit_intercept=True,
        max_iter=5,
    ):
        """
         Parameters:
         ----------
         convergence_tolerance : float
             The stopping criteria based on
             the value of the gradient.

         learning_rate : float
             Eta for the gradient update

         fit_intercept : bool
             If true, will add column of ones

         max_iter : int
             Stopping criteria based on the number of updates.

         """
        self.convergence_tolerance = convergence_tolerance
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter

    @staticmethod
    def get_gradient_update_term(X, y, weights):
        """
        Get the gradient update term.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        weights : np.ndarray
            Weights of Logistic Regression

        """
        return ((weights @ X.T) - y) @ X

    def fit(self, X, y):
        """
        Run the fitting procedure for the LogisticRegression

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix

        y : np.ndarray
            Target vector

        weights : np.ndarray
            Weights of Logistic Regression

        """
        # Add column of ones
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Initialize random weights
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=(1, X.shape[1]))

        # Get the gradient
        gradient = AdalineNetwork.get_gradient_update_term(
            X=X, y=y, weights=self.weights
        )

        iter_count = 1
        # While convergence criteria not met
        while (
            np.any(np.abs(gradient) > self.convergence_tolerance)
            and iter_count < self.max_iter
        ):
            # Update weights
            self.weights = self.weights - self.learning_rate * gradient

            # Calculate weights
            gradient = AdalineNetwork.get_gradient_update_term(
                X=X, y=y, weights=self.weights
            )

            # Increment coutn
            iter_count += 1

            # Stop if gradient explodes
            if pd.isnull(gradient).any():

                print("Exploding gradient")
                break

    def predict_prob(self, X):
        """
        Get output scores for predictions on X

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix
        """
        # Add ones if fit_intercept
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        # Make predictions
        return (self.weights @ X.T).reshape(-1)

    def predict(self, X):
        """
        Make predictions on X

        Parameters:
        ----------
        X : np.ndarray
            Feature matrix
        """
        # Round to 0 or 1
        return np.round(self.predict_prob(X=X))
