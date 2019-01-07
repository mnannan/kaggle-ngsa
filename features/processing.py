import numpy as np
import pandas as pd

MAX_VALUE = 1e9


def features_processing(x: pd.DataFrame) -> pd.DataFrame:
    """
    This is used before feeding data in a model.
    It replaces NaN by 0 and np.inf by MAX_VALUE and cast to float
    """
    x = x.fillna(0)
    for column in x.columns:
        if x[column].max() == np.inf:
            column_max_value = x[x[column] < np.inf][column].max()
            if column_max_value <= MAX_VALUE:
                x[column] = x[column].replace(np.inf, MAX_VALUE)
            else:
                print(f'{column} is larger than {MAX_VALUE}')
                x[column] = x[column].replace(np.inf, column_max_value * MAX_VALUE)
    return x.astype('float')
