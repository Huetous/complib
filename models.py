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
import catboost as cat
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
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

# class HuetousLowLGB (train, Dataset, cv)
# class HuetousLGB (fit, X,y)

class HuetousLGB(BaseEstimator):
    def __init__(self, params, task='reg', eval_metric='mae', need_proba=False,
                 n_est=5000, early_stopping_rounds=200,
                 verbose=500, n_fold=5):
        if task is 'reg':
            self.model = lgb.LGBMRegressor(**params, n_estimators=n_est, n_jobs=-1)
            self.eval_metric = eval_metric
        elif task is 'clf':
            self.model = lgb.LGBMClassifier(**params, n_estimators=n_est, n_jobs=-1)
            self.eval_metric = eval_auc
        self.task = task
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.need_proba = need_proba
        self.n_fold = n_fold

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.columns = X_tr.columns
        self.model.fit(X_tr, y_tr,
                       eval_set=[(X_tr, y_tr), (X_val, y_val)],
                       eval_metric=self.eval_metric,
                       verbose=self.verbose,
                       early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, X):
        if self.task is 'reg':
            return self.model.predict(X, num_iteration=self.model.best_iteration_)
        elif self.task is 'clf':
            if self.need_proba is True:
                return self.model.predict_proba(X, num_iteration=self.model.best_iteration_)[:, 1]
            else:
                return self.model.predict(X, num_iteration=self.model.best_iteration_)

    def feature_importance(self):
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = self.columns
        feature_importance["importance"] = self.model.feature_importances_
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:50].index
        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.show()

    # def add_feature_importance(self, fold_feature_importance):
    #     self.feature_importance = np.stack[self.feature_importance, fold_feature_importance]
    #
    # def get_feature_importance(self):
    #     return self.feature_importance / self.n_fold


class HuetousXGB(BaseEstimator):
    def __init__(self, params, task='reg', need_proba=False,
                 n_est=5000, early_stopping_rounds=200,
                 verbose=500):
        # if task is 'reg':
        #     self.eval_metric = eval_metric
        # elif task is 'clf':
        #     self.eval_metric = eval_metric
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

        self.params = params
        self.n_est = n_est
        self.need_proba = need_proba

    def fit(self, X_tr, y_tr, X_val, y_val):
        dtrain = xgb.DMatrix(data=X_tr, label=y_tr)
        dval = xgb.DMatrix(data=X_val, label=y_val)

        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(dtrain=dtrain,
                               num_boost_round=self.n_est, evals=watchlist,
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=self.verbose, params=self.params)

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
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
                 n_est=5000, early_stopping_rounds=200,
                 verbose=500):
        if task is 'reg':
            self.model = cat.CatBoostRegressor(**params, iterations=n_est,
                                               early_stopping_rounds=early_stopping_rounds,
                                               verbose=verbose)
        elif task is 'clf':
            self.model = cat.CatBoostClassifier(**params, iterations=n_est,
                                                early_stopping_rounds=early_stopping_rounds,
                                                verbose=verbose)

    def fit(self, X_tr, y_tr, X_val, y_val):
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
