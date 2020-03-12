import gc
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold


def _permutation_importance(X, y, estimator, metric):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series or numpy vector')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(estimator, BaseEstimator):
        raise TypeError('Parameter <estimator> must be an sklearn estimator')

    n_jobs = multiprocessing.cpu_count()

    if metric == 'r2':
        metric_func = lambda t, p: r2_score(t, p)
    elif metric == 'mse':
        metric_func = lambda t, p: -mean_squared_error(t, p)
    elif metric == 'mae':
        metric_func = lambda t, p: -mean_absolute_error(t, p)
    elif metric == 'accuracy' or metric == 'acc':
        metric_func = lambda t, p: accuracy_score(t, p)
    elif metric == 'auc':
        metric_func = lambda t, p: roc_auc_score(t, p)
    elif hasattr(metric, '__call__'):
        metric_func = metric
    else:
        raise ValueError('Parameter <metric> must be a metric string or a callable')

    base_score = metric_func(y, estimator.predict(X))

    def _permutation_importance(iC):
        tC = X[iC].copy()
        X[iC] = X[iC].sample(frac=1, replace=True).values
        shuff_score = metric_func(y, estimator.predict(X))
        X[iC] = tC.copy()
        del tC
        gc.collect()
        return base_score - shuff_score

    importances = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
    pool = ThreadPool(n_jobs)
    imps = pool.map(_permutation_importance, list(X.columns))
    for iC in range(len(imps)):
        importances.iloc[0, iC] = imps[iC]

    return importances


def _permutation_importance_cv(X, y, estimator, metric, n_splits=3, shuffle=True, seed=0):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame')
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas Series or numpy vector')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if not isinstance(estimator, BaseEstimator):
        raise TypeError('Parameter<estimator> must be an sklearn estimator')
    if not isinstance(n_splits, int):
        raise TypeError('Parameter<n_splits> must be an integer')
    if n_splits < 1:
        raise ValueError('Parameter<n_splits> must be 1 or greater')
    if not isinstance(shuffle, bool):
        raise TypeError('Parameter<shuffle> must be True or False')

    importances = pd.DataFrame(np.zeros((n_splits, X.shape[1])), columns=X.columns)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    iF = 0
    for train_ix, test_ix in kf.split(X):
        t_est = clone(estimator)
        t_est.fit(X.iloc[train_ix, :], y[train_ix])
        t_imp = _permutation_importance(X.iloc[test_ix, :].copy(),
                                        y[test_ix].copy(),
                                        t_est, metric)
        importances.loc[iF, :] = t_imp.loc[0, :]
        iF += 1

    return importances


def get_permutation_importance(importances, sort_by=np.mean, plot=True):
    df = pd.melt(importances, var_name='Feature', value_name='Importance')
    dfg = (df.groupby(['Feature'])['Importance']
           .aggregate(sort_by)
           .reset_index()
           .sort_values('Importance', ascending=False))
    if plot:
        sns.barplot(x='Importance', y='Feature', data=df, order=dfg['Feature'])
    return dfg
