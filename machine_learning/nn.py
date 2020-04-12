from toolz import pipe
import numpy as np


def softmax(y):
    return np.exp(y) / np.sum(np.exp(y), axis=1).reshape(-1, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SequentialNetwork:
    def __init__(
        self,
        *modules,
        learning_rate,
        convergence_tol,
        n_iter,
        batch_size,
        fit_intercept=True
    ):
        self.batch_size = batch_size
        self.convergence_tol = convergence_tol
        self.n_iter = n_iter
        self.lr = learning_rate
        self.fit_intercept = fit_intercept
        self.modules = list(modules)

    def fit(self, X, y):
        self.loss = []
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Add column of ones if fit_intercept
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        n_iter = 0

        while n_iter < 10 or (
            self.convergence_tol < np.mean(np.abs(np.diff(self.loss[-100:])))
            and n_iter < self.n_iter
        ):
            n_iter += 1
            batch_mask = np.random.choice(
                np.arange(X.shape[0]), replace=True, size=self.batch_size
            )

            preds = self(X[batch_mask])
            delta_list = self.get_delta_list(target=y[batch_mask])
            gradient_list = self.get_gradient_updates(delta_list=delta_list)
            loss = np.mean((y[batch_mask] - preds) ** 2)
            self.loss.append(loss)

    def __call__(self, X):
        return pipe(X, *tuple(self.modules))

    def get_delta_list(self, target):
        delta_list = []
        for module_num in range(len(self.modules) - 1, -1, -1):
            module_output = self.modules[module_num].get_previous_output()
            if module_num == len(self.modules) - 1:
                delta_list.append(
                    self.modules[module_num].get_last_layer_gradient(
                        target=target, output=softmax(module_output)
                    )
                )

            else:
                delta_list.insert(
                    0,
                    self.modules[module_num].gradient_update(
                        grad_accumulated=delta_list[0],
                        prev_weights=self.modules[module_num + 1].weight,
                    )[:, 1:],
                )
        return delta_list

    def get_gradient_updates(self, delta_list):
        gradients = []
        for module_num in range(len(self.modules) - 1, -1, -1):
            prev_input = self.modules[module_num].prev_input
            if self.modules[module_num].bias:
                prev_input = np.concatenate(
                    [np.ones((prev_input.shape[0], 1)), prev_input], axis=1
                )

            gradient = prev_input.T @ delta_list[module_num]

            if gradient.shape != self.modules[module_num].weight.shape:
                raise ValueError("Gradient just aint the right shape")
            self.modules[module_num].weight += self.lr * gradient
            gradients.append(gradient)
        return gradients

    def predict_prob(self, X):
        if self.fit_intercept:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        return self(X)

    def predict(self, X):
        predicted_probs = self.predict_prob(X)

        if predicted_probs.shape[1] == 1:
            return np.round(predicted_probs, 0)
        return np.argmax(predicted_probs, axis=1)


class LinearSigmoid:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(
            low=-0.01,
            high=0.01,
            size=(in_features + 1 if bias else in_features, out_features),
        )
        self.bias = bias

    def __call__(self, X):
        self.prev_input = X
        if self.bias:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        return sigmoid(X @ self.weight)

    def gradient_update(self, grad_accumulated, prev_weights):
        prev_output = self(self.prev_input)
        prev_output = np.concatenate(
            [np.ones((prev_output.shape[0], 1)), prev_output], axis=1
        )
        return (prev_output * (1 - prev_output)) * np.dot(
            grad_accumulated, prev_weights.T
        )

    def get_last_layer_gradient(self, output, target):
        # return ((target / output) - (1 - target) / (1 - output)) * output * (1 - output)
        return -2 * (output - target) * output * (1 - output)

    def get_previous_output(self):
        return self(self.prev_input)
