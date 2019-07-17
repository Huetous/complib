import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin


class TargetSmoothedEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, alpha=5):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.alpha = 5

    def fit(self, X, y):
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype) == 'object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column', col, ' not in X')

        # Encode each element of each column
        self.maps = dict()
        for col in self.cols:
            global_mean = X[col].mean()

            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                nrows = y[X[col] == unique].count()
                mean = y[X[col] == unique].mean()
                tmap[unique] = (mean * nrows + global_mean * self.alpha) / (nrows + self.alpha)
            self.maps[col] = tmap

        return self

    def transform(self, X, y=None):
        res = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col] == val] = mean_target
            res[col] = vals
        return res

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class TargetEncoderCV(TargetEncoder):
    def __init__(self, n_splits=3, shuffle=True, cols=None, seed=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.cols = cols
        self.seed = 0

    def fit(self, X, y):
        self.target_encoder = TargetSmoothedEncoder(cols=self.cols).fit(X, y)
        return self

    def transform(self, X, y=None):
        # Use target encoding from fit() if this is test data
        if y is None:
            return self.target_encoder.transform(X)

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
        res = X.copy()
        for tr_idx, val_idx in kf.split(X):
            te = TargetEncoder(cols=self.cols).fit(X.iloc[tr_idx, :], y.iloc[tr_idx])
            res.iloc[val_idx, :] = te.transform(X.iloc[val_idx, :])
        return res

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
