from toolz import pipe, dicttoolz
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Callable, Union
import requests
import io
from scipy.stats import mode
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

"""
This sections contains the code for the 
feedforward network
"""


def softmax(y):
    """Numerically stable softmax function"""
    y = y - np.max(y, axis=1).reshape(-1, 1)
    return np.exp(y) / np.sum(np.exp(y), axis=1).reshape(-1, 1)


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


class SequentialNetwork:
    """
    This class contains the mechanism to train a
    feedforward network.
    """

    def __init__(self, *modules, learning_rate, convergence_tol, n_iter, batch_size):
        """
        Parameters:
        -----------
        *modules : LinearSigmoid
            An arbitrary number of network units composed
            of a linear transformation and a sigmoid activation.

        learning_rate : float
            eta for gradient descent

        convergence_tol : float
            Training will stop if the change in loss drops
            below this.

        n_iter : int
            The maximum number of training iterations.

        batch_size : int
            The number of examples used to calculate
            the gradient over at each step
        """
        self.batch_size = batch_size
        self.convergence_tol = convergence_tol
        self.n_iter = n_iter
        self.lr = learning_rate
        self.modules = list(modules)

    def fit(self, X, y):
        """
        Function to fit the network to set
        of inputs.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix

        y : np.ndarray
            The target vector
        """
        self.loss = []

        # Reshape y if needed (for compatibility with
        # other tools
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_iter = 0
        # While stopping criteria unmet...
        while n_iter < 10 or (
            self.convergence_tol < np.mean(np.abs(np.diff(self.loss[-100:])))
            and n_iter < self.n_iter
        ):
            n_iter += 1
            # Get a random batch
            batch_mask = np.random.choice(
                np.arange(X.shape[0]), replace=True, size=self.batch_size
            )

            # Make predictions over the batch
            preds = self(X[batch_mask])

            # Calculate the delta
            delta_list = self.get_delta_list(target=y[batch_mask])

            # Make the gradient updates
            gradient_list = self.get_gradient_updates(delta_list=delta_list)

            # Calculate and track the loss
            loss = np.mean((y[batch_mask] - preds) ** 2)
            self.loss.append(loss)

    def __call__(self, X):
        """
        Forward pass applied the units to the inputs.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of inputs
        """
        return pipe(X, *tuple(self.modules))

    def get_delta_list(self, target):
        """
        Function to get the deltas for all
        the layers

        Parameters:
        -----------
        target : np.ndarray
            Class labels (onehot encoded)
        """
        delta_list = []

        # Iterate through the units
        for module_num in range(len(self.modules) - 1, -1, -1):
            # Get the most recent output of the module
            module_output = self.modules[module_num].get_previous_output()

            # If it's the last layer in the network
            # get the output layer delta value
            if module_num == len(self.modules) - 1:
                delta_list.append(
                    self.modules[module_num].get_last_layer_gradient(
                        target=target, output=softmax(module_output)
                    )
                )

            # If it's not the last unit...
            else:
                # Get the delta value for the unit
                delta_list.insert(
                    0,
                    self.modules[module_num].gradient_update(
                        grad_accumulated=delta_list[0],
                        prev_weights=self.modules[module_num + 1].weight,
                    )[:, 1:],
                )
        return delta_list

    def get_gradient_updates(self, delta_list):
        """
        Function to make the gradient updates. Happens in place.

        Parameters:
        -----------
        delta_list : list
            The list of deltas calculated above
        """
        # Iterate through the modules
        for module_num in range(len(self.modules) - 1, -1, -1):
            # Get the previous input of the unit
            prev_input = self.modules[module_num].prev_input

            # If the unit has a bias, add a column of ones
            if self.modules[module_num].bias:
                prev_input = np.concatenate(
                    [np.ones((prev_input.shape[0], 1)), prev_input], axis=1
                )

            # Calculate the gradient from delta and the previous input
            gradient = prev_input.T @ delta_list[module_num]

            # Make the update to the weights
            self.modules[module_num].weight += self.lr * gradient

    def predict_prob(self, X):
        """
        Function to get the raw output of the multi-net

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix to make predictions over
        """
        return self(X)

    def predict(self, X):
        """
        Function to get the predicted class labels.

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix to make predictions over.
        """
        predicted_probs = self.predict_prob(X)

        # Get the largest output value as the label
        return np.argmax(predicted_probs, axis=1)


class LinearSigmoid:
    """
    Class that makes up the basic unit of the feedforward
    network above.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Parameters:
        -----------
        in_features : int
            The number of features as inputs

        out_features : int
            The number of features as outputs.

        bias : bool
            If true, a bias column of ones will be added.
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize random weights
        self.weight = np.random.uniform(
            low=-0.01,
            high=0.01,
            size=(in_features + 1 if bias else in_features, out_features),
        )
        self.bias = bias

    def __call__(self, X):
        """
        Make a forward pass through the network

        Parameters:
        -----------
        X : np.ndarray
            The feature matrix
        """
        # Keep track of what went in
        # so we can calculate the delta
        self.prev_input = X

        if self.bias:
            # Add column of ones if creating bias unit
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        # Calculate the output
        return sigmoid(X @ self.weight)

    def gradient_update(self, grad_accumulated, prev_weights):
        """
        Get the non-last layer gradient update
        values:

        Parameters:
        -----------
        grad_accumulated : np.ndarray
            Delta of the downstream unit

        prev_weights : np.ndarray
            Weights of the downstream unit

        """
        # Get previous output
        prev_output = self(self.prev_input)

        # Add ones if bias
        if self.bias:
            prev_output = np.concatenate(
                [np.ones((prev_output.shape[0], 1)), prev_output], axis=1
            )

        # Get O (1 - O) * delta @ W_i+1
        return (prev_output * (1 - prev_output)) * np.dot(
            grad_accumulated, prev_weights.T
        )

    def get_last_layer_gradient(self, output, target):
        """
        Function to get the delta value
        for the last unit in the network

        Parameters:
        -----------
        output : np.ndarray
            Matrix of outputs from multi-net

        target : np.ndarray
            Matrix of one-hot class labels
        """
        # (O - Y) O (1 - O)
        return -2 * (output - target) * output * (1 - output)

    def get_previous_output(self):
        """
        Helper to get the output given the
        previous input
        """
        return self(self.prev_input)


"""
This section has code for normalizing the data
"""


class Standardizer:
    def __init__(self, mean=True, std=True):
        self.mean = mean
        self.std = std

    def fit(self, X):
        if self.mean:
            self.df_means = X.mean(axis=0)
        if self.std:
            self.df_std = X.std(axis=0)

    def transform(self, X):
        if not self.mean and not self.std:
            return X
        if self.mean:
            df_xf = X - self.df_means
        if self.std:
            non_zero = np.bitwise_not(np.isclose(self.df_std, 0))
            df_xf = np.where(non_zero, df_xf / self.df_std, df_xf)

        return df_xf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


"""
This section has code for cross-validation and 
grid searching
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


"""
This section has the code for the actual experiments
"""


def score_func(y, yhat):
    return np.mean(np.argmax(y, axis=1) == yhat)


def run_classification_experiment(
    X,
    y,
    learning_rate_choices=np.linspace(0.001, 0.01, 10),
    hidden_layer_choices=list(range(3, 16, 3)),
    n_iter=10000,
    conv_tol=0.001,
    filename=None,
):
    kfold = KFoldStratifiedCV(num_folds=5)
    accuracy_0h = []
    accuracy_1h = []
    accuracy_2h = []
    baseline = []

    split = 0
    for train, test in kfold.split(X=X, y=y.reshape(-1,)):
        split += 1
        logger.info(f"CV Iteration: {split}")
        logger.info("Standardizing data")
        max_scaler = Standardizer()
        X_train = max_scaler.fit_transform(X[train])
        X_test = max_scaler.transform(X[test])
        y_train = pd.get_dummies(y[train]).values

        if split == 1:
            logger.info("Finding learning rate for H0")
            h0_callable = lambda lr: SequentialNetwork(
                LinearSigmoid(
                    in_features=X_train.shape[1], out_features=y_train.shape[1]
                ),
                convergence_tol=conv_tol,
                n_iter=n_iter,
                learning_rate=lr,
                batch_size=48,
            )

            results = list(
                GridSearchCV(
                    model_callable=h0_callable,
                    param_grid={"lr": learning_rate_choices},
                    scoring_func=score_func,
                    cv_object=KFoldCV(num_folds=3),
                ).get_cv_scores(X=X_train, y=y_train)
            )

            best_model_h0 = h0_callable(
                **sorted(results, key=lambda x: x[-1], reverse=True)[0][0]
            )
            logger.info(
                f"Results: {sorted(results, key=lambda x: x[-1], reverse=True)[0][0]}"
            )

            logger.info("Finding topology and learning rate for H1")
            h1_callable = lambda h1, lr: SequentialNetwork(
                LinearSigmoid(in_features=X_train.shape[1], out_features=h1, bias=True),
                LinearSigmoid(in_features=h1, out_features=y_train.shape[1], bias=True),
                convergence_tol=conv_tol,
                n_iter=n_iter,
                learning_rate=lr,
                batch_size=X_train.shape[1],
            )

            results = list(
                GridSearchCV(
                    model_callable=h1_callable,
                    param_grid={
                        "h1": hidden_layer_choices,
                        "lr": learning_rate_choices,
                    },
                    scoring_func=score_func,
                    cv_object=KFoldCV(num_folds=3),
                ).get_cv_scores(X=X_train, y=y_train)
            )

            best_model_h1 = h1_callable(
                **sorted(results, key=lambda x: x[-1], reverse=True)[0][0]
            )
            logger.info(
                f"Results: {sorted(results, key=lambda x: x[-1], reverse=True)[0][0]}"
            )

            logger.info("Finding topology and learning rate for H2")
            h2_callable = lambda h1, h2, lr: SequentialNetwork(
                LinearSigmoid(in_features=X_train.shape[1], out_features=h1, bias=True),
                LinearSigmoid(in_features=h1, out_features=h2, bias=True),
                LinearSigmoid(in_features=h2, out_features=y_train.shape[1], bias=True),
                convergence_tol=conv_tol,
                n_iter=n_iter,
                learning_rate=lr,
                batch_size=X_train.shape[1],
            )

            results = list(
                GridSearchCV(
                    model_callable=h2_callable,
                    param_grid={
                        "h1": hidden_layer_choices,
                        "h2": hidden_layer_choices,
                        "lr": learning_rate_choices,
                    },
                    scoring_func=score_func,
                    cv_object=KFoldCV(num_folds=3),
                ).get_cv_scores(X=X_train, y=y_train)
            )
            logger.info(
                f"Results: {sorted(results, key=lambda x: x[-1], reverse=True)[0][0]}"
            )
            best_model_h2 = h2_callable(
                **sorted(results, key=lambda x: x[-1], reverse=True)[0][0]
            )

        best_model_h0.fit(X_train, y_train)
        best_model_h1.fit(X_train, y_train)
        best_model_h2.fit(X_train, y_train)

        if split == 1 and filename:
            logger.info(f"Creating prediction output files: {filename}")
            preds = pd.DataFrame(
                np.hstack(
                    [
                        X_test,
                        y[test].reshape(-1, 1),
                        np.array(best_model_h0.predict(X_test)).reshape(-1, 1),
                        np.array(best_model_h1.predict(X_test)).reshape(-1, 1),
                        np.array(best_model_h2.predict(X_test)).reshape(-1, 1),
                    ]
                )
            )
            orig_cols = list(preds.columns)
            orig_cols[-4:] = [
                "actuals",
                "h0_prediction",
                "h1_predictions",
                "h2_predictions",
            ]
            preds.columns = orig_cols
            preds.to_csv(filename, index=False)

        baseline.append(np.mean(mode(y[train]).mode[0] == y[test]))
        accuracy_0h.append(np.mean(best_model_h0.predict(X_test) == y[test]))
        accuracy_1h.append(np.mean(best_model_h1.predict(X_test) == y[test]))
        accuracy_2h.append(np.mean(best_model_h2.predict(X_test) == y[test]))
    return {
        "models": {"h0": best_model_h0, "h1": best_model_h1, "h2": best_model_h2,},
        "accuracy": {
            "h0": accuracy_0h,
            "h1": accuracy_1h,
            "h2": accuracy_2h,
            "baseline": baseline,
        },
    }


"""
Load the data and run the experiments
"""
if __name__ == "__main__":
    np.random.seed(73)
    logger.info("Iris Experiment")
    iris_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )

    X, y = (
        iris_data.drop(["class"], axis=1).values,
        iris_data["class"].astype("category").cat.codes.values,
    )
    iris_results = run_classification_experiment(
        X=X,
        y=y,
        learning_rate_choices=list(np.linspace(0.01, 0.8, 5)),
        hidden_layer_choices=[3, 4],
        n_iter=500,
        conv_tol=0.001,
        filename="iris_prediction_results.csv",
    )

    logger.info(f"Iris Results: {dicttoolz.valmap(np.mean, iris_results['accuracy'])}")

    logger.info("Running glass experiment")

    glass_data = pd.read_csv(
        io.BytesIO(
            requests.get(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
                verify=False,
            ).content
        ),
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
        glass_data["class"].astype("category").cat.codes.values,
    )

    glass_results = run_classification_experiment(
        X,
        y=y,
        learning_rate_choices=np.linspace(0.025, 0.1, 5),
        hidden_layer_choices=[3, 5, 7, 8],
        n_iter=1000,
        conv_tol=0.0001,
        filename="glass_prediction_results.csv",
    )
    logger.info(
        f"Glass Results: {dicttoolz.valmap(np.mean, glass_results['accuracy'])}"
    )

    logger.info("Running soybean experiment")

    # Next, we repeat this process on the Soybean data
    soybean_data = pipe(
        pd.read_csv(
            io.BytesIO(
                requests.get(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
                    verify=False,
                ).content
            ),
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

    np.random.seed(73)

    soybean_results = run_classification_experiment(
        X,
        y=y,
        learning_rate_choices=np.linspace(0.003, 0.005, 3),
        hidden_layer_choices=list(range(5, 11, 2)),
        filename="soybean_prediction_results.csv",
    )
    logger.info(
        f"Soybean Results: {dicttoolz.valmap(np.mean, soybean_results['accuracy'])}"
    )

    np.random.seed(73)
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

    house_votes_results = run_classification_experiment(
        X=X,
        y=y,
        learning_rate_choices=list(np.linspace(0.001, 0.01, 10)),
        hidden_layer_choices=[4, 7, 9],
        n_iter=1500,
        conv_tol=0.01,
        filename="house_votes_results.csv",
    )

    logger.info(
        f"House Results: {dicttoolz.valmap(np.mean, house_votes_results['accuracy'])}"
    )

    logger.info("Running breast cancer classification experiment")

    # Load and clean the data
    breast_cancer = (
        pd.read_csv(
            io.BytesIO(
                requests.get(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                    verify=False,
                ).content
            ),
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
        .dropna()
    )

    # Run the experiment
    X, y = (
        breast_cancer.drop(["id_number", "class"], axis=1).values,
        breast_cancer["class"].astype("category").cat.codes.values.reshape(-1,),
    )
    np.random.seed(73)
    breast_cancer_results = run_classification_experiment(
        X=X, y=y, filename="breast_cancer_predictions.csv"
    )

    logger.info(
        f"Breast Cancer Results: {dicttoolz.valmap(np.mean, breast_cancer_results['accuracy'])}"
    )
