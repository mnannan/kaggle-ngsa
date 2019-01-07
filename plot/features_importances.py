from typing import List

import numpy as np
import pandas as pd


def plot_features_importance(features_importance: np.ndarray, features_name: List):
    features_importance = pd.Series(features_importance, index=features_name)
    features_importance = features_importance.sort_values(ascending=False)
    features_importance.plot(kind='barh')
    print(features_importance)
