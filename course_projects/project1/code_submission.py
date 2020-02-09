"""
Course: Introduction to Machine Learning
Assignment: Project 1
Author: David Atlas
"""
import pandas as pd
from typing import Callable
import numpy as np
from scipy import stats
from toolz import pipe
from collections import Counter, OrderedDict
import logging

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


"""
This section contains some utilities for model building. 
1. Function for one-hot encoding continous variables.
2. Confusion Matrix function
3. KFoldCV - Class for KFold cross validation
4. MulticlassClassifier - Creates one vs. rest classifiers (useful for Winnow2 and boolean outputs
"""


def discretize_dataframe(df: pd.DataFrame, discretize_boundries: dict):
    df = df.assign(
        **{
            col: pd.cut(df[col], bins=boundaries)
            for col, boundaries in discretize_boundries.items()
        }
    )
    return pd.get_dummies(
        data=df, columns=[c for c in df.columns if "class" not in c], drop_first=True
    )


def confusion_matrix(actuals, predictions):
    """Function to find the confusin matrix for a classifier"""

    # Get all the class values
    actual_values = np.sort(np.unique(actuals))

    # Get the dataframe where columns are y and rows are yhat
    return pd.DataFrame(
        {
            y: [
                np.sum((actuals == y) & (predictions == yhat)) for yhat in actual_values
            ]
            for y in actual_values
        },
        index=actual_values,
    )


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


class MulticlassClassifier:
    """
    Class to do one vs. rest multiclass classification using
    Boolean output classifier.

    """

    def __init__(self, model_cls: Callable, classes: np.ndarray, cls_kwargs):
        """
        Parameters
        ----------
        model_cls : Callable
            A callable that returns the model object to use in fitting.

        classes : np.ndarray
            An array containing the values in `y` for which to create a classifier.

        cls_kwargs : dict
            A dictionary of args for `model_cls` mapping the class value
            to a dictionary of kwargs.
        """
        self.classes = classes
        # Create the models (mapping from class to model)
        self.models = {
            element: model_cls(**cls_kwargs.get(element)) for element in self.classes
        }

    @staticmethod
    def _get_y_binary(y, cls):
        # Transform multivalued outputs into one vs. rest booleans
        # where `cls` is the value of 1.
        return np.where(y == cls, 1, 0)

    def fit(self, X, y):
        """
        Fit the classifiers across all the models.
        """
        if set(y) - set(self.classes):
            raise ValueError("y contains elements not in `classes`")

        for cls, model in self.models.items():
            # Create the binary response for `cls`
            y_binary = MulticlassClassifier._get_y_binary(y, cls)
            # Fit the the model for that class.
            model.fit(X, y_binary)

    def predict(self, X):
        """
        Gets the highest probability class across all the one vs. rest classifiers.
        """
        # Get the prediction_prob across all the classes.
        predictions = {cls: model.predict_prob(X) for cls, model in self.models.items()}

        # Get the class corresponding to the largest probability.
        return [
            max(predictions.keys(), key=lambda x: predictions[x][prediction])
            for prediction in range(X.shape[0])
        ]


def fit_predict_kfold(model_obj, X, y, kfold, randomseed=None):
    # Set seed if not None
    if randomseed:
        np.random.seed(randomseed)

    logger.setLevel(logging.CRITICAL)
    iteration = 0
    # Iterate through splits
    for train, test in kfold.split(X=X):
        iteration += 1

        if iteration == kfold.num_folds:
            logger.setLevel(logging.INFO)

        # Fit model
        model_obj.fit(X[train, :], y[train])
        # make predictions on test set
        predicted = model_obj.predict(X[test, :])
        actuals = y[test]
        # Log confusion matrix for last fold
        if iteration == kfold.num_folds:
            logger.info(confusion_matrix(actuals=actuals, predictions=predicted))


"""
This section contains the implementation of our algorithms.
- Winnow2
- Naive Bayes
"""


class Winnow2:
    def __init__(
        self,
        weight_scaler: float,
        threshold: float,
        num_features: int,
        verbose: bool = True,
    ):
        self.weight_scaler = weight_scaler
        self.threshold = threshold
        self.num_features = num_features
        self.weights = np.ones((num_features,))
        self.verbose = verbose

    def predict_prob(self, X: np.array):
        """
        Function to get the raw prediction score (not binary)
        """
        return X @ self.weights

    def predict(self, X: np.array):
        """
        Function to get the binary prediction value.
        """
        return self.predict_prob(X) > self.threshold

    def adjust_weights(self, X: np.array, scale_func: Callable):
        """
        Function to either promote or demote, based on whether division or
        multiplication is passed as the scaling function.
        """
        if isinstance(X, list):
            X = np.array(X)

        if self.verbose:
            logger.info(f"Initial weights: {self.weights}")
            logger.info(f"Training instance: {X}")

        new_weights = np.where(
            X == 1, scale_func(self.weights, self.weight_scaler), self.weights
        )

        if self.verbose:
            logger.info(f"Updated weights: {new_weights}")

        return new_weights

    def promote_weights(self, X: np.array):
        if self.verbose:
            logger.info("Promoting weights...")
        return self.adjust_weights(X=X, scale_func=np.multiply)

    def demote_weights(self, X: np.array):
        if self.verbose:
            logger.info("Demoting weights...")
        return self.adjust_weights(X=X, scale_func=np.true_divide)

    def run_training_iteration(self, X: np.array, y: bool):
        """
        Runs a single training iteration for Winnow2.
        """
        yhat = self.predict(X)
        if self.verbose:
            logger.info(f"Actual: {y} Prediction: {yhat}")

        # If prediction is correct, do nothing
        if yhat == y:
            if self.verbose:
                logger.info("Correct prediction. No updates.")
            return

        # If prediction is 0 and y is 1, promote
        if not yhat and y:
            self.weights = self.promote_weights(X)
            return

        # If prediction is 1 and y is 0, demote
        self.weights = self.demote_weights(X)
        return

    def fit(self, X, y):
        for X_instance, y_instance in zip(X, y):
            self.run_training_iteration(X_instance, y_instance)


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
                    f"Class: {k} Mean: {np.round(v.mean(), 2)} Std: {np.round(v.std(), 2)}"
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
                logger.info(f"Class: {k} p: {np.round(v.p, 2)}")
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


"""
This section contains the actual experiments. We begin by running our 3 algorithm variants 
over the Iris dataset.

To reiterate the report, our 3 algorithm variants are 
1. Winnow2 with discretized one-hot encoded features.
2. Naive Bayes with the same discretized one-hot encoded features.
3. Naive Bayes with discrete multivalued and continuous inputs.
"""

# We read in the data
iris_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)

# We discretize the data. See the report for the rationale here.
X_iris, y_iris = discretize_dataframe(
    iris_data,
    discretize_boundries={
        "sepal_length": [0, 5.5, 10],
        "sepal_width": [0, 3, 10],
        "petal_length": [0, 1, 1.6, 10],
        "petal_width": [0, 2, 5, 10],
    },
).pipe(
    lambda df: (
        df.filter(like="]").values,
        df["class"].astype("category").cat.codes.values,
    )
)

# We create the model object and KFoldCV object
iris_winnow2 = MulticlassClassifier(
    model_cls=Winnow2,
    classes=[0, 1, 2],
    cls_kwargs={
        cls: {"weight_scaler": 2, "threshold": 1, "num_features": X_iris.shape[1]}
        for cls in [0, 1, 2]
    },
)

kfold = KFoldCV(num_folds=5, shuffle=True)

logger.info("Fitting Winnow2 on Boolean Iris Dataset")
fit_predict_kfold(
    model_obj=iris_winnow2, X=X_iris, y=y_iris, kfold=kfold, randomseed=73
)

logger.info("Fitting Naive Bayes on Boolean Iris Dataset")
iris_naive_bayes = MulticlassClassifier(
    model_cls=NaiveBayes,
    classes=[0, 1, 2],
    cls_kwargs={
        cls: {
            "column_distribution_map": {
                col: "multinomial" for col in range(X_iris.shape[1])
            },
            "binomial": True,
        }
        for cls in [0, 1, 2]
    },
)

fit_predict_kfold(
    model_obj=iris_naive_bayes, X=X_iris, y=y_iris, kfold=kfold, randomseed=73
)

logger.info("Fitting Naive Bayes on Continuous Iris Dataset")
iris_naive_bayes_continuous = NaiveBayes(
    column_distribution_map={col: "gaussian" for col in range(X_iris.shape[1])}
)

fit_predict_kfold(
    model_obj=iris_naive_bayes_continuous,
    X=iris_data.drop("class", axis=1).values,
    y=iris_data["class"].astype("category").cat.codes.values,
    kfold=kfold,
    randomseed=73,
)

# Next, we fit the models over the cancer dataset. We read in the data.
# We drop the NULL values, as there are only 16 of them (with 699 total rows)
logger.info("Fitting Cancer data classifiers")
cancer_data = (
    pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        header=None,
        names=[
            "id_number",
            "clump_thickness",
            "uniformity_cell_size",
            "uniformity_cell_shape",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitosis",
            "class",
        ],
    )
    .replace("?", np.NaN)
    .dropna(axis=0, how="any")
    .astype("int")
    .assign(instance_class=lambda df: df["class"].astype("category").cat.codes)
    .drop(["class", "id_number"], axis=1)
)

cancer_data_boolean = pipe(
    cancer_data,
    lambda df: pd.get_dummies(
        data=df,
        columns=[col for col in df.columns if col != "instance_class"],
        drop_first=True,
    ),
)

logger.info("Fitting Winnow2 on Boolean Cancer")
cancer_winnow2 = Winnow2(
    weight_scaler=2, threshold=1, num_features=cancer_data_boolean.shape[1] - 1
)
fit_predict_kfold(
    model_obj=cancer_winnow2,
    X=cancer_data_boolean.drop("instance_class", axis=1).values,
    y=cancer_data_boolean["instance_class"].values,
    kfold=kfold,
    randomseed=73,
)

logger.info("Fitting Naive Bayes on Boolean Cancer")
cancer_naive_bayes_bool = NaiveBayes(
    column_distribution_map={
        col_idx: "multinomial" for col_idx in range(cancer_data_boolean.shape[1] - 1)
    }
)

fit_predict_kfold(
    model_obj=cancer_naive_bayes_bool,
    X=cancer_data_boolean.drop("instance_class", axis=1).values,
    y=cancer_data_boolean["instance_class"].values,
    kfold=kfold,
    randomseed=73,
)

logger.info("Fitting Naive Bayes on Multinomial Cancer")
cancer_naive_bayes = NaiveBayes(
    column_distribution_map={
        col_idx: "multinomial"
        for col_idx in range(cancer_data.drop("instance_class", axis=1).shape[1])
    }
)

fit_predict_kfold(
    model_obj=cancer_naive_bayes,
    X=cancer_data.drop("instance_class", axis=1).subtract(1).values,
    y=cancer_data["instance_class"].values,
    kfold=kfold,
    randomseed=73,
)

# Next, we run the same process over the glass dataset. Again, see the report for the
# rationale on the discretization.

glass_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    header=None,
    names=[
        "id_number",
        "refractive_index",
        "sodium",
        "magnesium",
        "aluminum",
        "silicon",
        "potassium",
        "calcium",
        "barium",
        "iron",
        "class",
    ],
)

discretize_boundries_glass = {
    "refractive_index": [1.518],
    "sodium": [12.5, 14],
    "magnesium": [1, 2, 3.25],
    "aluminum": [1.2, 1.5, 2],
    "silicon": [72.6, 72.8, 73.2, 73.4],
    "potassium": [0.2, 0.4, 0.5, 0.6],
    "calcium": [8.6, 9, 10],
    "barium": [0.35],
    "iron": [0.2, 0.6],
}

X_glass, y_glass = discretize_dataframe(
    glass_data.drop("id_number", axis=1), discretize_boundries_glass
).pipe(
    lambda df: (
        df.drop("class", axis=1).values,
        df["class"].astype("category").cat.codes.values,
    )
)

logger.info("Fitting Winnow2 on Boolean Glass")
glass_winnow2 = MulticlassClassifier(
    model_cls=Winnow2,
    classes=[0, 1, 2, 3, 4, 5],
    cls_kwargs={
        cls: {"weight_scaler": 2, "threshold": 1, "num_features": X_glass.shape[1]}
        for cls in [0, 1, 2, 3, 4, 5]
    },
)

fit_predict_kfold(
    model_obj=glass_winnow2, X=X_glass, y=y_glass, kfold=kfold, randomseed=74
)

logger.info("Fitting Naive Bayes on Boolean Glass")
glass_naive_bayes_bool = MulticlassClassifier(
    model_cls=NaiveBayes,
    classes=[0, 1, 2, 3, 4, 5],
    cls_kwargs={
        cls: {
            "column_distribution_map": {
                col: "multinomial" for col in range(X_glass.shape[1])
            },
            "binomial": True,
        }
        for cls in [0, 1, 2, 3, 4, 5]
    },
)

fit_predict_kfold(
    model_obj=glass_naive_bayes_bool, X=X_glass, y=y_glass, kfold=kfold, randomseed=74
)

logger.info("Fitting Naive Bayes on Continuous Glass")
X_glass_continuous = glass_data.drop(["id_number", "class"], axis=1).values
y_glass_continuous = glass_data["class"].astype("category").cat.codes.values

glass_naive_bayes = NaiveBayes(
    column_distribution_map={
        col: "gaussian" for col in range(X_glass_continuous.shape[1])
    },
)

fit_predict_kfold(
    model_obj=glass_naive_bayes,
    X=X_glass_continuous,
    y=y_glass_continuous,
    kfold=kfold,
    randomseed=74,
)


# Next, we repeat this process on the Soybean data
soybean_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
    header=None,
    names=[
        "date",
        "plant-stand",
        "precip",
        "temp",
        "hail",
        "crop-hist",
        "area-damaged",
        "severity",
        "seed-tmt",
        "germination",
        "plant-growth",
        "leaves",
        "leafspots-halo",
        "leafspots-marg",
        "leafspot-size",
        "leaf-shread",
        "leaf-malf",
        "leaf-mild",
        "stem",
        "lodging",
        "stem-cankers",
        "canker-lesion",
        "fruiting-bodies",
        "external decay",
        "mycelium",
        "int-discolor",
        "sclerotia",
        "fruit-pods",
        "fruit spots",
        "seed",
        "mold-growth",
        "seed-discolor",
        "seed-size",
        "shriveling",
        "roots",
        "class",
    ],
)
