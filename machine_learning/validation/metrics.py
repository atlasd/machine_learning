import numpy as np
import pandas as pd


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


def accuracy(actuals, predictions):
    return np.mean(actuals == predictions)


def mean_squared_error(actuals, predictions):
    return np.mean((actuals - predictions) ** 2)
