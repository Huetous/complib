from xgboost import XGBRegressor
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
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from huetous.metrics import eval_auc

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')


# class huetousLinear():

# Linear
# Trees
# NN
# Exotic (FM/FFM)

# -------------------------------------------------------------------------------------
class HueLGB:
    def __init__(self, params=None, task='regression', eval_metric=None):
        if task == 'regression':
            if params is not None:
                self.model = lgb.LGBMRegressor(**params, n_jobs=-1)
            else:
                self.model = lgb.LGBMRegressor(n_jobs=-1)
            if eval_metric is None:
                self.eval_metric = 'mae'
        else:
            if params is not None:
                self.model = lgb.LGBMClassifier(**params, n_jobs=-1)
            else:
                self.model = lgb.LGBMClassifier(n_jobs=-1)
            if eval_metric is None:
                self.eval_metric = eval_auc

    def train(self, X_tr, y_tr, X_val, y_val):
        self.X_tr_columns = X_tr.columns

        self.model.fit(X_tr, y_tr,
                       eval_set=[(X_tr, y_tr), (X_val, y_val)],
                       eval_metric=self.eval_metric,
                       verbose=10000,
                       early_stopping_rounds=200)

    def predict(self, X):
        return self.model.predict(X, self.model.best_iteration_)

    def feature_importance(self):
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = self.X_tr_columns
        feature_importance["importance"] = self.model.feature_importances_
        return feature_importance


# -------------------------------------------------------------------------------------
class HueXGB:
    def __init__(self, params=None, num_rounds=50,
                 early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_rounds = num_rounds
        self.early_stopping_round = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.X_tr_columns = X_tr.columns

        dtrain = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)
        watchlist = [(dtrain, 'train'), (dval, 'val')]

        self.model = xgb.train(params=self.params,
                               dtrain=dtrain,
                               num_boost_round=self.num_rounds,
                               early_stopping_rounds=self.early_stopping_round,
                               evals=watchlist,
                               verbose_eval=self.verbose_eval)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X, feature_names=X.columns)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)

    def feature_importance(self):
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = self.X_tr_columns
        feature_importance["importance"] = self.model.feature_importances_
        return feature_importance


# -------------------------------------------------------------------------------------
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


class HueCatBoost:
    def __init__(self, params=None, task='regression'):
        if task == 'regression':
            if params is not None:
                self.model = CatBoostRegressor(**params, eval_metric='MAE')
            else:
                self.model = CatBoostRegressor(eval_metric='MAE')
        else:
            if params is not None:
                self.model = CatBoostClassifier(**params, eval_metric='AUC')
            else:
                self.model = CatBoostClassifier(eval_metric='AUC')

    def train(self, X_tr, y_tr, X_val, y_val):
        self.model.fit(X_tr, y_tr,
                       eval_set=(X_val, y_val),
                       cat_features=[],
                       use_best_model=True,
                       verbose=False)

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
