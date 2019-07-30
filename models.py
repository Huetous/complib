from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from tensorflow.python.keras import models, layers
import numpy as np
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.base import clone
from sklearn.base import BaseEstimator
from huelib.metrics import eval_auc
import gc
import time

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')

# class huetousLinear():

# Linear
# Trees
# NN
# Exotic (FM/FFM)

# -------------------------------------------------------------------------------------
base_lgb_params = {
    # regression, huber, binary, multiclass, xentropy
    'objective': 'regression',

    # mae, mse, rmse, huber, auc, binary_logloss,
    # multi_logloss, binary_error, multi_error, cross_entropy,
    'metric': 'rmse',

    # Used in binary classification
    # weight of labels with positive class
    # 'scale_pos_weight': 1,
}

lgb_params = {'num_leaves': 256,
              'min_child_samples': 79,
              'objective': 'binary',
              'max_depth': 13,
              'learning_rate': 0.03,
              "subsample_freq": 3,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'auc',
              "verbosity": -1,
              'reg_alpha': 0.3,
              'reg_lambda': 0.3,
              'colsample_bytree': 0.9
              }


class HuetousLGB(BaseEstimator):
    def __init__(self, params, task='reg', eval_metric='mae', need_proba=False,
                 n_estimators=1000, early_stopping_rounds=200,
                 verbose=100):
        if task is 'reg':
            self.estim = lgb.LGBMRegressor
            self.eval_metric = eval_metric
        elif task is 'clf':
            self.estim = lgb.LGBMClassifier
            self.eval_metric = eval_auc
        self.params = params
        self.n_jobs = -1
        self.n_estimators = n_estimators
        self.task = task
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.need_proba = need_proba

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.estim = self.estim(**self.params, n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        self.estim.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  eval_metric=self.eval_metric,
                  verbose=self.verbose,
                  early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, X):
        if self.task is 'clf' and self.need_proba is True:
            return self.estim.predict_proba(X=X, num_iteration=self.estim.best_iteration_)[:, 1]
        else:
            return self.estim.predict(X=X, num_iteration=self.estim.best_iteration_)


class HuetousXGB(BaseEstimator):
    def __init__(self, params,
                 num_boost_round=5000, early_stopping_rounds=200,
                 verbose=500):
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.params = params
        self.num_boost_round = num_boost_round

    def fit(self, X_tr, y_tr, X_val, y_val):
        dtrain = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_tr.columns)

        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(dtrain=dtrain,
                               num_boost_round=self.num_boost_round, evals=watchlist,
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=self.verbose, params=self.params)

    def predict(self, X):
        dtest = xgb.DMatrix(data=X, feature_names=X.columns)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)


cat_params = {
    # verbose: N - every N iters show log
    # more iters (default=1000)
    #
    # cat_features_names = X.columns # here we specify names of categorical features
    # cat_features = [X.columns.get_loc(col) for col in cat_features_names]
    # cat(cat_features)
    #
    # 'early_stopping_rounds': 200, if metric doesn`t improve for N rounds than stop training
    # we dont need to wait all 1000
    #
    # faster trainig
    # 'task_type': 'GPU',
    # 'border_count': 32,
    # use_best_model=True
}


class HuetousCatBoost(BaseEstimator):
    def __init__(self, params, task='reg', eval_metric='MAE',
                 iterations=5000, early_stopping_rounds=200,
                 verbose=500):
        if task is 'reg':
            self.model = cat.CatBoostRegressor(**params, iterations=iterations,
                                               early_stopping_rounds=early_stopping_rounds,
                                               eval_metric=eval_metric,
                                               # loss_function=eval_metric,
                                               verbose=verbose)
        elif task is 'clf':
            self.model = cat.CatBoostClassifier(**params, iterations=iterations,
                                                early_stopping_rounds=early_stopping_rounds,
                                                eval_metric=eval_metric,
                                                # loss_function=eval_metric,
                                                verbose=verbose)
        self.verbose = verbose

    def fit(self, X_tr=None, y_tr=None, X_val=None, y_val=None):
        self.model.fit(X_tr, y_tr,
                       eval_set=(X_val, y_val),
                       cat_features=[],
                       use_best_model=True,
                       verbose=self.verbose)

    def predict(self, X):
        return self.model.predict(X)


# -------------------------------------------------------------------------------------
# Train separate models for each type
def do_sep_models_train(model, params, X_tr, y_tr, X_te, target_name, type):
    if not isinstance(type, str):
        raise ValueError('Parameter <type> should be string.')

    S_train = pd.DataFrame({'index': list(X_tr.index),
                            type: X_tr[type].values,
                            'oof': [0] * len(X_tr),
                            'target': y_tr.values})
    S_test = pd.DataFrame(
        {'index': list(X_te.index),
         type: X_te[type].values,
         'preds': [0] * len(X_te)})

    for t in X_tr[type].unique():
        print(f'Training of type {t}')
        X_t = X_tr.loc[X_tr[type] == t]
        X_test_t = X_te.loc[X_te[type] == t]
        y_t = X_tr.loc[X_tr[type] == t, target_name]

        model = model(**params).fit(X_t, y_t, X_test_t)
        preds = model.predict()
        S_train.loc[S_train['type'] == t, 'oof'] = preds['oof']
        S_test.loc[S_test['type'] == t, 'preds'] = preds['preds']

    return [S_train, S_test]
