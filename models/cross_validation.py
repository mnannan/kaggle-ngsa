from typing import Dict

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

seed = 123


def cross_validation(Model, model_args: Dict, x: np.ndarray, y: np.ndarray,
                     n_splits: int =5, seed: int = seed):
    """
    Run cross validation on a model and a training set
    :param Model: Model on which we want to apply cross validation
    :param model_args: Dict of arguments of the function
    e.g: model_args = {
        'n_estimators': 100,
        'max_depth': 3,
        'random_state': seed
    }
    :param x: training set
    :param y: labels
    :param n_splits: number of splits for the cross validation
    :param seed: seed to reproduce results
    :return: predicted vectors, list of errors during the cross validation
    """
    f1_list = []

    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    y_pred = np.zeros(y.shape)
    i = 1
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Model(**model_args)
        model.fit(X_train, y_train)

        y_pred[test_index] = model.predict(X_test)

        f1_list.append(f1_score(y_test, y_pred[test_index]))

        print(f'Epoch {i}')
        print(f'f1: {f1_list[i-1]}')
        i += 1
    return y_pred, f1_list
