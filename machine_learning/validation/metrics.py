import numpy as np
import pandas as pd


def confusion_matrix(actuals, predictions):
    actual_values = np.sort(np.unique(actuals))
    prediction_values = np.sort(np.unique(predictions))

    return pd.DataFrame(
        {
            y: [
                np.sum((actuals == y) & (predictions == yhat))
                for yhat in prediction_values
            ]
            for y in actual_values
        },
        index=prediction_values,
    )
