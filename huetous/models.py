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
from catboost import CatBoostRegressor
from sklearn.base import clone

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')


# class huetousLGB():
# class huetousXGB():
# class huetousLinear():
# class huetousTrees():

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
class hueLGB(metaModel):
    def __init__(self, params=None, task='regression', eval_metric='mae'):
        if task == 'regression':
            super().__init__(lgb.LGBMRegressor, params, mean_absolute_error)
        # else:
        # super().__init__(lgb.LGBMClassifier, params, log_loss)

    def train(self, X_tr, y_tr, X_val, y_val):
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
        feature_importance["feature"] = X.columns
        feature_importance["importance"] = model.feature_importances_
        return feature_importance


# -------------------------------------------------------------------------------------
class hueXGB(metaModel):
    def __init__(self, params=None, task='regression'):
        super().__init__(None, params)
        self.task = task

    def train(self, X_tr, y_tr, X_val=None, y_val=None):
        train_data = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        val_data = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)
        # clone?
        self.model = xgb.train(dtrain=train_data, num_boost_round=self.n_estimators,
                               evals=[(train_data, 'train'), (val_data, 'val')],
                               early_stopping_rounds=200,
                               verbose_eval=500, params=self.params)

    def predict(self, X):
        # clone?
        return self.model.predict(xgb.DMatrix(X, feature_names=X.columns),
                                  ntree_limit=self.model.best_ntree_limit)


# -------------------------------------------------------------------------------------
class hueCatBoost():
    def __init__(self, params=None):
        self.model = CatBoostRegressor(**params)

    def train(self, X_tr, y_tr, X_val, y_val):
        self.model.fit(X_tr, y_tr,
                       eval_set=(X_val, y_val),
                       cat_features=[],
                       use_best_model=True,
                       verbose=False)

    def predict(self, X):
        return self.model.predict(X)


# -------------------------------------------------------------------------------------
class hueRidgeCV():
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
class hueSklearn(object):
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

# Linear
# Trees
# NN
# Exotic (FM/FFM)

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
