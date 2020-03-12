from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier, XGBRegressor
from sklearn import model_selection
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
import pandas as pd
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------------------------
class Blending(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# --------------------------------------------------------------------------------------------
class Stacking:
    def __init__(self, base_models, meta_model, n_folds=5, shuffle=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.shuffle = shuffle

    def fit(self, X, y):
        kfold = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=156)

        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        oof_pred = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for tr_idx, te_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)

                instance.fit(X[tr_idx], y[tr_idx])
                y_pred = instance.predict(X[te_idx])

                oof_pred[te_idx, i] = y_pred

        self.meta_model_.fit(oof_pred, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# --------------------------------------------------------------------------------------------
class VotingClassifier:
    def __init__(self, models, voting_type='hard'):
        if voting_type not in ['hard', 'soft']:
            raise ValueError('Parameter <type> incorrectly specified.')
        self.voter = ensemble.VotingClassifier(estimators=[clone(x) for x in models], voting=type)

    def fit(self, X_tr, y_tr):
        self.voter.fit(X_tr, y_tr)

    def predict(self, X_te):
        return self.voter.predict(X_te)
