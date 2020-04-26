from toolz import pipe
import numpy as np


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
            gradient_list = self.get_gradient_updates(
                delta_list=delta_list, verbose=True if n_iter < 4 else False
            )

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

    def get_gradient_updates(self, delta_list, verbose=False):
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
            if verbose:
                print("Gradient update: ")
                print(gradient)
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
