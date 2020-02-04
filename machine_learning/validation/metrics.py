import numpy as np
import pandas as pd


def confusion_matrix(actuals, predictions):
    actual_values = np.sort(np.unique(actuals))

    return pd.DataFrame(
        {
            y: [
                np.sum((actuals == y) & (predictions == yhat)) for yhat in actual_values
            ]
            for y in actual_values
        },
        index=actual_values,
    )
