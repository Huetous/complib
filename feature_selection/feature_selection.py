from sklearn import feature_selection, model_selection
from sklearn.cluster import KMeans
from sklearn.base import clone
from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb

import statsmodels.api as sm
from tqdm import tqdm


# --------------------------------------------------------------------------------------------

# Performs recursive feature elimination
def do_rfe(model, X, y,
           cv_split=None, scoring='accuracy',
           random_state=42):
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=random_state)

    selector = feature_selection.RFECV(clone(model), step=1, scoring=scoring, cv=cv_split)
    selector.fit(X, y)
    print('do_feat_rfe: Done')
    return X.columns.values[selector.get_support()]


# Performs feature selection by BorutaPy selector
def do_boruta(model, X, y,
              max_iter=500,
              random_state=42):
    selector = BorutaPy(clone(model), n_estimators='auto', verbose=0,
                        random_state=random_state, max_iter=max_iter)
    selector.fit(X.values, y.values)
    print('do_feat_boruta: Done')
    return X.columns.values[selector.support_]


# Performs principal component analysis
def do_pca(df, threshold=0.99):
    pca = PCA().fit(df)
    explained_variance = 0.0
    components = 0

    for var in pca.explained_variance_ratio_:
        explained_variance += var
        components += 1
        if explained_variance >= threshold:
            break
    print('Explained_variance: {},\n Components: {}'.format(explained_variance, components))
    print('do_feat_pca: Done')
    return PCA(n_components=components).fit_transform(df)


# Selects important features from model
def do_sel_from_model(X, y, model=None,
                      params=None, n_estimators=1000,
                      threshold='1.25*median'):
    if params is None:
        raise ValueError('Parameter <params> must be specified.')
    if model is None:
        model = lgb.LGBMClassifier(**params,
                                   n_estimators=n_estimators,
                                   n_jobs=-1)

    selector = SelectFromModel(model, threshold=threshold)
    selector.fit(X, y)
    features = X.loc[:, selector.get_support()].columns.tolist()
    print(str(len(features)), 'selected features')
    print('do_feat_from_model: Done')
    return features

