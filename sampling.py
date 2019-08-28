from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def do_smote(X, y, seed=42, sampling_strategy=0.2):
    sm = SMOTE(random_state=seed, sampling_strategy=sampling_strategy)
    X_sm, y_sm = sm.fit_sample(X, y)
    X_sm = pd.DataFrame(X_sm, columns=X.columns)

    return X_sm, y_sm


def do_ros(X, y, seed=42, sampling_strategy=0.2):
    ros = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
    X_ros, y_ros = ros.fit_sample(X, y)
    X_ros = pd.DataFrame(X_ros, columns=X.columns)
    return X_ros, y_ros


def do_rus(X, y, seed=42, sampling_strategy='majority'):
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)
    X_rus, y_rus = rus.fit_sample(X, y)
    X_rus = pd.DataFrame(X_rus, columns=X.columns)
    return X_rus, y_rus


def do_augment(x, y, t=2):
    xp, xn = [], []
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xp.append(x1)

    for i in range(t // 2):
        mask = y == 0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:, c] = x1[ids][:, c]
        xn.append(x1)

    xp = np.vstack(xp)
    xn = np.vstack(xn)
    yp = np.ones(xp.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xp, xn])
    y = np.concatenate([y, yp, yn])
    return x, y
