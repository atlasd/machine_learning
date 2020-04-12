import numpy as np
import pandas as pd
import logging
from typing import Callable
from collections import Counter
from toolz import pipe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()

"""
Code for the cross validation.
"""


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


"""
This part is the code for a one vs. rest classifier 
"""


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


"""
This part is the two linear discriminant 
algorithm implementations.
"""


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


def mode(y):
    """
    Function to get the mode of an array
    """
    return Counter(y).most_common(1)[0][0]


class MaxScaler:
    """
    Class that scales everything to [-1, 1] interval.
    """

    def fit(self, X):
        # Get the max values
        self.maxes = np.abs(X).max()

    def transform(self, X):
        # Scale by said values
        return X / self.maxes

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


"""
Run the experiments
"""
if __name__ == "__main__":
    np.random.seed(73)

    logger.info("Running Breast Cancer Experiment")
    breast_cancer = (
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
        .astype("float", errors="ignore")
        .dropna(how="any", axis=0)
    )

    ms = MaxScaler()

    X, y = (
        breast_cancer.drop(["id_number", "class"], axis=1).values,
        breast_cancer["class"].astype("category").cat.codes.values,
    )

    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_adaline = []
    accuracy_lr = []
    baseline = []

    # Stratifier KFOLD CV
    for train, test in kfold.split(X=X, y=y):
        sweet_adaline = AdalineNetwork(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=100,
            learning_rate=0.0001,
        )

        logistic_regression = LogisticRegressionClassifier(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=100,
            learning_rate=0.0001,
        )

        # Append the baseline accuracy (mode of training target)
        baseline.append(np.mean(mode(y[train]) == y[test]))

        # Fit the models on the transformed training data
        sweet_adaline.fit(ms.fit_transform(X[train]), y[train])
        logistic_regression.fit(ms.fit_transform(X[train]), y[train])

        # Append the accuracy
        accuracy_adaline.append(
            np.mean(sweet_adaline.predict(ms.transform(X[test])) == y[test])
        )
        accuracy_lr.append(
            np.mean(logistic_regression.predict(ms.transform(X[test])) == y[test])
        )

    logger.info(f"Baseline Accuracy: {np.mean(baseline)}")
    logger.info(f"Adaline Accuracy: {np.mean(accuracy_adaline)}")
    logger.info(f"Logistic Regression Accuracy: {np.mean(accuracy_lr)}")

    # Glass Experiment
    logger.info("Running Glass Experiment")
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

    X, y = (
        glass_data.drop(["id_number", "class"], axis=1).values,
        glass_data["class"].astype("category").cat.codes,
    )

    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_adaline = []
    accuracy_lr = []
    baseline = []

    # Stratifier KFOLD CV
    for train, test in kfold.split(X=X, y=y):
        sweet_adaline = MulticlassClassifier(
            model_cls=lambda *args: AdalineNetwork(
                convergence_tolerance=0.0001,
                fit_intercept=True,
                max_iter=5000,
                learning_rate=0.005,
            ),
            classes=np.unique(y),
            cls_kwargs={i: {} for i in np.unique(y)},
        )

        logistic_regression = LogisticRegressionClassifier(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=15000,
            learning_rate=0.005,
        )

        ms = MaxScaler()

        # Fit the models
        sweet_adaline.fit(ms.fit_transform(X[train]), y[train])
        logistic_regression.fit(ms.fit_transform(X[train]), y[train])

        # Append results
        baseline.append(np.mean(mode(y[train]) == y[test]))
        accuracy_adaline.append(
            np.mean(sweet_adaline.predict(ms.transform(X[test])) == y[test])
        )
        accuracy_lr.append(
            np.mean(logistic_regression.predict(ms.transform(X[test])) == y[test])
        )

    logger.info(f"Baseline Accuracy: {np.mean(baseline)}")
    logger.info(f"Adaline Accuracy: {np.mean(accuracy_adaline)}")
    logger.info(f"Logistic Regression Accuracy: {np.mean(accuracy_lr)}")

    logger.info("Running Soybean Experiment")

    # Next, we repeat this process on the Soybean data
    soybean_data = pipe(
        pd.read_csv(
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
                "instance_class",
            ],
        )
        .pipe(
            lambda df: df.loc(axis=1)[df.nunique() > 1]
        )  # drop columns with no variance
        .assign(
            instance_class=lambda df: df["instance_class"].astype("category").cat.codes
        )
    )

    X, y = (
        pd.get_dummies(
            soybean_data.drop("instance_class", axis=1),
            columns=soybean_data.drop("instance_class", axis=1).columns,
            drop_first=True,
        ).values,
        soybean_data["instance_class"].values,
    )

    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_adaline = []
    accuracy_lr = []
    baseline = []

    # Stratified KFold CV
    for train, test in kfold.split(X=X, y=y):
        sweet_adaline = MulticlassClassifier(
            model_cls=lambda *args: AdalineNetwork(
                convergence_tolerance=0.0001,
                fit_intercept=True,
                max_iter=1000,
                learning_rate=0.001,
            ),
            classes=np.unique(y),
            cls_kwargs={i: {} for i in np.unique(y)},
        )

        logistic_regression = LogisticRegressionClassifier(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=1000,
            learning_rate=0.001,
        )

        # Fit Adaline
        sweet_adaline.fit(X[train], y[train])

        # Fit Logistic Regression
        logistic_regression.fit(X[train], y[train])

        # Append accuracy
        baseline.append(np.mean(mode(y[train]) == y[test]))
        accuracy_adaline.append(np.mean(sweet_adaline.predict(X[test]) == y[test]))
        accuracy_lr.append(np.mean(logistic_regression.predict(X[test]) == y[test]))

    logger.info(f"Baseline Accuracy: {np.mean(baseline)}")
    logger.info(f"Adaline Accuracy: {np.mean(accuracy_adaline)}")
    logger.info(f"Logistic Regression Accuracy: {np.mean(accuracy_lr)}")

    # Iris Experiment
    logger.info("Running Iris Experiment")

    iris_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )

    X, y = (
        iris_data.drop(["class"], axis=1).values,
        iris_data["class"].astype("category").cat.codes,
    )

    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_adaline = []
    accuracy_lr = []
    baseline = []

    # Stratified KFold CV
    for train, test in kfold.split(X=X, y=y):
        sweet_adaline = MulticlassClassifier(
            model_cls=lambda *args: AdalineNetwork(
                convergence_tolerance=0.0001,
                fit_intercept=True,
                max_iter=1000,
                learning_rate=0.005,
            ),
            classes=np.unique(y),
            cls_kwargs={i: {} for i in np.unique(y)},
        )

        logistic_regression = LogisticRegressionClassifier(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=1000,
            learning_rate=0.005,
        )

        ms = MaxScaler()

        # Fit models
        sweet_adaline.fit(ms.fit_transform(X[train]), y[train])
        logistic_regression.fit(ms.fit_transform(X[train]), y[train])

        # Append results
        baseline.append(np.mean(mode(y[train]) == y[test]))
        accuracy_adaline.append(
            np.mean(sweet_adaline.predict(ms.transform(X[test])) == y[test])
        )
        accuracy_lr.append(
            np.mean(logistic_regression.predict(ms.transform(X[test])) == y[test])
        )

    logger.info(f"Baseline Accuracy: {np.mean(baseline)}")
    logger.info(f"Adaline Accuracy: {np.mean(accuracy_adaline)}")
    logger.info(f"Logistic Regression Accuracy: {np.mean(accuracy_lr)}")

    # House Votes Experiment
    logger.info("Running House Votes Experiment")
    house_votes_data = pipe(
        pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
            header=None,
            names=[
                "instance_class",
                "handicapped-infants",
                "water-project-cost-sharing",
                "adoption-of-the-budget-resolution",
                "physician-fee-freeze",
                "el-salvador-aid",
                "religious-groups-in-schools",
                "anti-satellite-test-ban",
                "aid-to-nicaraguan-contras",
                "mx-missile",
                "immigration",
                "synfuels-corporation-cutback",
                "education-spending",
                "superfund-right-to-sue",
                "crime",
                "duty-free-exports",
                "export-administration-act-south-africa",
            ],
        )
        .replace("?", np.NaN)
        .replace("y", 1)
        .replace("n", 0),
        lambda df: pd.get_dummies(
            df, columns=df.columns, drop_first=True, dummy_na=True
        ),
    )

    X, y = (
        house_votes_data.drop(
            ["instance_class_republican", "instance_class_nan"], axis=1
        ).values,
        house_votes_data["instance_class_republican"].values,
    )

    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_adaline = []
    accuracy_lr = []
    baseline = []
    for train, test in kfold.split(X=X, y=y):
        sweet_adaline = AdalineNetwork(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=1000,
            learning_rate=0.0001,
        )

        logistic_regression = LogisticRegressionClassifier(
            convergence_tolerance=0.0001,
            fit_intercept=True,
            max_iter=1000,
            learning_rate=0.0001,
        )

        # Fit the models
        sweet_adaline.fit(X[train], y[train])
        logistic_regression.fit(X[train], y[train])

        # Append the results

        baseline.append(np.mean(mode(y[train]) == y[test]))
        accuracy_adaline.append(np.mean(sweet_adaline.predict(X[test]) == y[test]))
        accuracy_lr.append(np.mean(logistic_regression.predict(X[test]) == y[test]))

    logger.info(f"Baseline Accuracy: {np.mean(baseline)}")
    logger.info(f"Adaline Accuracy: {np.mean(accuracy_adaline)}")
    logger.info(f"Logistic Regression Accuracy: {np.mean(accuracy_lr)}")
