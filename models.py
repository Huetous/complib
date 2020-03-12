import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


class HuetousLogReg(BaseEstimator):
    def __init__(self, params, need_proba=False,
                 max_iter=100, eval_metric=None):
        if eval_metric is None:
            raise ValueError('Parameter <eval_metric> must be specified.')
        self.clf = LogisticRegression(**params, max_iter=max_iter)
        self.eval_metric = eval_metric
        self.need_proba = need_proba

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.clf.fit(X_tr, y_tr)
        if self.need_proba:
            y_pred = self.clf.predict_proba(X_val)[:, 1]
        else:
            y_pred = self.clf.predict(X_val)
        print(f'LogReg {self.eval_metric.__class__.__name__}: {self.eval_metric(y_val, y_pred)}')

    def predict(self, X):
        if self.need_proba:
            return self.clf.predict_proba(X)[:, 1]
        else:
            return self.clf.predict(X)


class HuetousLGB(BaseEstimator):
    def __init__(self, params, task='reg', eval_metric='mae', need_proba=False,
                 n_estimators=1000, early_stopping_rounds=200,
                 verbose=100):
        if task is 'reg':
            self.clf = lgb.LGBMRegressor
            self.eval_metric = eval_metric
        elif task is 'clf':
            self.clf = lgb.LGBMClassifier
            self.eval_metric = roc_auc_score
        self.params = params
        self.n_jobs = -1
        self.n_estimators = n_estimators
        self.task = task
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.need_proba = need_proba

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.clf = self.clf(**self.params,
                            n_estimators=self.n_estimators,
                            n_jobs=self.n_jobs)
        self.clf.fit(X_tr, y_tr,
                     eval_set=[(X_tr, y_tr), (X_val, y_val)],
                     eval_metric=self.eval_metric,
                     verbose=self.verbose,
                     early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, X):
        if self.task is 'clf' and self.need_proba is True:
            return self.clf.predict_proba(X=X, num_iteration=self.clf.best_iteration_)[:, 1]
        else:
            return self.clf.predict(X=X, num_iteration=self.clf.best_iteration_)


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


class HuetousCatBoost(BaseEstimator):
    def __init__(self, params, task='reg', eval_metric='MAE',
                 iterations=5000, early_stopping_rounds=200,
                 verbose=500):
        if task is 'reg':
            self.model = cat.CatBoostRegressor(**params, iterations=iterations,
                                               early_stopping_rounds=early_stopping_rounds,
                                               eval_metric=eval_metric,
                                               verbose=verbose)
        elif task is 'clf':
            self.model = cat.CatBoostClassifier(**params, iterations=iterations,
                                                early_stopping_rounds=early_stopping_rounds,
                                                eval_metric=eval_metric,
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
# Train separate models for each categorical value in given column
def do_sep_models_train(model, params, X_tr, y_tr, X_te, target_name, c):
    if not isinstance(c, str):
        raise ValueError('Parameter <type> should be string.')

    S_train = pd.DataFrame({'index': list(X_tr.index),
                            c: X_tr[c].values,
                            'oof': [0] * len(X_tr),
                            'target': y_tr.values})
    S_test = pd.DataFrame(
        {'index': list(X_te.index),
         c: X_te[c].values,
         'preds': [0] * len(X_te)})

    for t in X_tr[c].unique():
        print(f'Training of type {t}')
        X_t = X_tr.loc[X_tr[c] == t]
        X_test_t = X_te.loc[X_te[c] == t]
        y_t = X_tr.loc[X_tr[c] == t, target_name]

        model = model(**params).fit(X_t, y_t, X_test_t)
        preds = model.predict()
        S_train.loc[S_train['type'] == t, 'oof'] = preds['oof']
        S_test.loc[S_test['type'] == t, 'preds'] = preds['preds']

    return [S_train, S_test]
