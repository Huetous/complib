from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


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
