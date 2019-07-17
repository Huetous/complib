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
from huetous.utils import eval_auc

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')


# class huetousLGB():
# class huetousXGB():
# class huetousLinear():
# class huetousTrees():

# Linear
# Trees
# NN
# Exotic (FM/FFM)

# -------------------------------------------------------------------------------------
class metaModel():
    def __init__(self, model, params=None, metric=None):
        self.model = clone(model(**params), safe=False)
        self.metric = metric

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, params):
        self.model.set_params(params)

    def get_params(self):
        return self.model.get_params()

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_score(self, y_pred, y_true):
        return self.metric(y_true, y_pred)


# -------------------------------------------------------------------------------------
class huetousSklearn():
    def __init__(self, model, seed=0, params=None):
        params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    #
    # def fit(self, x, y):
    #     return self.model.fit(x, y)

    def feature_importances(self, x, y):
        self.feature = self.model.fit(x, y).feature_importances_.tolist()
        return self.feature

    def feature_importance_plotly(self, features, title):
        trace = go.Scatter(
            y=self.feature,
            x=features,
            mode='markers',
            marker=dict(
                sizemode='diameter',
                sizeref=1,
                size=25,
                color=self.feature,
                colorscale='Portland',
                showscale=True
            ),
            text=features
        )
        layout = go.Layout(
            autosize=True,
            title=title,
            hovermode='closest',
            yaxis=dict(
                title='Feature Importance',
                ticklen=5,
                gridwidth=2
            ),
            showlegend=True
        )
        fig = go.Figure(data=[trace], layout=layout)
        py.plot(fig, filename=title, auto_open=True)


# -------------------------------------------------------------------------------------
class huetousLGB(BaseEstimator):
    def __init__(self, params=None, task='regression', eval_metric=None):
        if task == 'regression':
            self.model = lgb.LGBMRegressor(**params)
            if eval_metric is None:
                self.eval_metric = 'mae'
        else:
            self.model = lgb.LGBMClassifier(**params, n_jobs=-1)
            self.eval_metric = eval_auc

    def train(self, X_tr, y_tr, X_val, y_val):
        self.X_tr_columns = X_tr.columns
        self.model.fit(X_tr, y_tr,
                       eval_set=[(X_tr, y_tr), (X_val, y_val)],
                       eval_metric=self.eval_metric,
                       verbose=10000,
                       early_stopping_rounds=200)

    def predict(self, X, best_iter=None):
        if best_iter:
            return self.model.predict(X, num_iteration=best_iter)
        else:
            return self.model.predict(X)

    def feature_importances(self):
        feature_importance = pd.DataFrame()
        feature_importance["feature"] = self.X_tr_columns
        feature_importance["importance"] = self.model.feature_importances_
        return feature_importance


# -------------------------------------------------------------------------------------
class huetousXGB(BaseEstimator):
    def __init__(self, params=None, num_rounds=50,
                 early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_rounds = num_rounds
        self.early_stopping_round = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        if X_val is not None:
            dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)
            watchlist = [(dtrain, 'train'), (dval, 'val')]
            self.model = xgb.train(params=self.params,
                                   dtrain=dtrain,
                                   num_boost_round=self.num_rounds,
                                   early_stopping_rounds=self.early_stopping_round,
                                   evals=watchlist,
                                   verbose_eval=self.verbose_eval)
        else:
            self.model = xgb.train(params=self.params,
                                   dtrain=dtrain,
                                   num_boost_round=self.num_rounds,
                                   early_stopping_rounds=self.early_stopping_round)
        return

    def predict(self, X):
        dtest = xgb.DMatrix(X, feature_names=X.columns)
        preds = self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)
        return preds


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
class huetousCatBoost():
    def __init__(self, params=None, task='regression'):
        if task == 'regression':
            self.model = CatBoostRegressor(**params, eval_metric='MAE')
        else:
            self.model = CatBoostClassifier(**params, eval_metric='AUC')

    def train(self, X_tr, y_tr, X_val, y_val):
        self.model.fit(X_tr, y_tr,
                       eval_set=(X_val, y_val),
                       cat_features=[],
                       use_best_model=True,
                       verbose=False)

    def predict(self, X):
        return self.model.predict(X)


# -------------------------------------------------------------------------------------
class huetousRidgeCV():
    def __init__(self, params=None):
        self.model = RidgeCV(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print('RidgeCV. Alpha: ', self.model.alpha_)

    def predict(self, X_val):
        return self.model.predict(X_val).reshape(-1, )

    def score(self, y_pred_val, y_val):
        return mean_absolute_error(y_val, y_pred_val)


# -------------------------------------------------------------------------------------
# Train separate models for each type
def do_sep_models_train(model, params, X_tr, y_tr, X_te, target_name, type):
    S_train = pd.DataFrame({'ind': list(X_tr.index),
                            'type': X_tr[type].values,
                            'oof': [0] * len(X_tr),
                            'target': y_tr.values})
    S_test = pd.DataFrame(
        {'ind': list(X_te.index),
         'type': X_te['type'].values,
         'predictions': [0] * len(X_te)})

    for t in X_tr[type].unique():
        print(f'Training of type {t}')
        X_t = X_tr.loc[X_tr[type] == t]
        X_test_t = X_te.loc[X_te[type] == t]
        y_t = X_tr.loc[X_tr[type] == t, target_name]
        model = model(**params).fit(X_t, y_t, X_test_t)
        preds = model.predict()
        S_train.loc[S_train['type'] == t, 'oof'] = preds['oof']
        S_test.loc[S_test['type'] == t, 'predictions'] = preds['prediction']

    return [S_train, S_test]


# Simple algo
# df
# test
# num_val
# val = df[:num_val]
# train = df[num_val:]
#
# model.fit(train)
# val_score = model.evaluate(val)
#
# model.fit(np.concatenate([train,val]))
# test_score = model.evaluate(test)

# Regul
# from keras import regularizers
# model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))
# Every coefficient in weight matrix will add 0.001*weight_coef_value
# in general value of loss

def conv2d_bin(input_shape=(150, 150, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model
