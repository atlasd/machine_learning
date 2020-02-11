import numpy as np
from scipy import stats
from toolz import pipe
from collections import Counter, OrderedDict
import logging

logger = logging.getLogger(__name__)


class NaiveBayes:
    def __init__(
        self,
        column_distribution_map: dict,
        alpha: float = 1,
        binomial: bool = False,
        verbose: bool = True,
    ):
        """
        Class to fit Naive Bayes.

        Parameters:
        -----------
        column_distribution_map : dict
            A dictionary that maps each column index to either
            "gaussian" or "multinomial". This will indicate which
            distribution each column should be fitted to.

        alpha : float
            This is the smoothing parameter alpha for multinomial
            distribution.s

        binomial : bool
            If the output is a binomial (boolean), set to true.
            This is only really used to produce proper prediction probabilities
            for the multiclass classification class.
        """
        self.binomial = binomial
        self.column_distribution_map = column_distribution_map
        self.fitted_distributions = {}
        self.is_fitted = False
        self.alpha = alpha
        self.verbose = verbose

    def _fit_gaussian(self, X, col_idx, y):
        """
        Fits classwise Gaussian distributions to `X[:, col_idx]`
        using the sample parameter MLEs.

        Parameters
        ----------
        X : np.ndarray
            Matrix of features.

        col_idx : int
            The column index for the column to fit the Gaussian to.

        y : np.ndarray
            Vector of target classes
        """
        # Dictionary to map each value in `y` to a Gaussian.
        gaussian_fits = {
            val: stats.norm(
                loc=X[y == val, col_idx].mean(),  # Class sample mean
                scale=max(X[y == val, col_idx].std(), 0.00001),  # Class sample std
            )
            for val in sorted(set(y))
        }
        if self.verbose:
            logger.info(f"Fitted Gaussians for column {col_idx}")
            for k, v in gaussian_fits.items():
                logger.info(
                    f"Class: {k} Mean: {np.round(v.mean())} Std: {np.round(v.std())}"
                )

        return gaussian_fits

    def _fit_multinomial(self, X, col_idx, y):
        """
        Fits classwise multinomial distributions to `X[:, col_idx]`
        using the sample parameter MLEs.

        Parameters
        ----------
        X : np.ndarray
            Matrix of features.

        col_idx : int
            The column index for the column to fit the multinomial to.

        y : np.ndarray
            Vector of target classes
        """
        fitted_distributions = {}
        all_X_values = list(range(int(X[:, col_idx].max()) + 1))
        # For each class...
        for val in sorted(set(y)):
            n = np.sum(y == val)  # Number of instances in the class
            relevant_subset = X[y == val, col_idx]  # Rows in X belonging to class
            value_counts = Counter(
                relevant_subset
            )  # Counts of the values in X in the class
            all_x_value_counts_smoothed = OrderedDict(
                {
                    x_val: self.alpha  # Just alpha if no values
                    if x_val not in value_counts
                    else value_counts[x_val]
                    + self.alpha  # Alpha + Num value occurences otherwise
                    for x_val in all_X_values  # across the values in the column of X
                }
            )
            # n + Alpha * m
            normalizer = n + self.alpha * len(all_X_values)

            # Create the distribution for each class.
            fitted_distributions[val] = stats.multinomial(
                n=n, p=np.array(list(all_x_value_counts_smoothed.values())) / normalizer
            )

        if self.verbose:
            logger.info(f"Fitted multinomials for column {col_idx}")
            for k, v in fitted_distributions.items():
                logger.info(f"Class: {k} p: {np.round(v.p)}")
        return fitted_distributions

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the classifier across all classes.
        """

        # For each feature column index in X
        for col_idx in range(X.shape[1]):
            if col_idx not in self.column_distribution_map:
                raise ValueError(f"No distribution given for column {col_idx}")

            # If the column has a multinomial tag, fit a multinomial.
            if self.column_distribution_map[col_idx] == "multinomial":
                self.fitted_distributions[col_idx] = self._fit_multinomial(
                    X=X, col_idx=col_idx, y=y
                )
            # Otherwise fit a Gaussian
            elif self.column_distribution_map[col_idx] == "gaussian":
                self.fitted_distributions[col_idx] = self._fit_gaussian(
                    X=X, col_idx=col_idx, y=y
                )

        self.is_fitted = True
        # The prior P(C) gets set to multinomial with p as the
        # proportion of observations in each class C
        self.prior = stats.multinomial(
            n=len(y), p=[np.sum(y == val) / len(y) for val in sorted(set(y))]
        )

    def _predict_one_class(self, X: np.ndarray, class_idx: int):
        """
        Generate prediction value for one class.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix
        class_idx : int
            The index of the class to get prediction value for.

        The output here is the production across features for a given class
        """
        return (
            np.array(
                [
                    self.fitted_distributions[col_idx][class_idx].pdf(
                        X[:, col_idx]
                    )  # get PDF if Gaussian
                    if self.column_distribution_map[col_idx] == "gaussian"
                    else self.fitted_distributions[col_idx][class_idx].p[
                        X[:, col_idx].astype("int")  # get p if multinomial
                    ]
                    for col_idx in range(X.shape[1])  # For each column in X
                ]
            ).prod(axis=0)
            * self.prior.p[class_idx]
        )

    def predict_prob(self, X):
        """
        Get the prediction probability for each row in X, for each class in y.
        """
        if not self.is_fitted:
            raise ValueError("Must fit model before predictions can be made")

        return pipe(
            [
                self._predict_one_class(
                    X=X, class_idx=class_idx
                )  # Get one class prediction
                for class_idx in self.fitted_distributions[0].keys()  # For each class
            ],
            np.vstack,  # Create a matrix where each row is prob of column being class
            # If self.binomial, return prob of C == 1, else return all rows.
            # Primarily for the multiclass classifier class.
            lambda arr: arr[1] if self.binomial else arr,
        )

    def predict(self, X):
        # Get the class prediction (argmax across classes)
        return np.argmax(self.predict_prob(X), axis=0)
