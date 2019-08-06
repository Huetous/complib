from sklearn import feature_selection, model_selection
from sklearn.cluster import KMeans
from sklearn.base import clone
from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.plotly as py
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from scipy.stats import ks_2samp
import statsmodels.api as sm
from tqdm import tqdm

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')


# --------------------------------------------------------------------------------------------
def do_rfe(model, X_train, y_train, cv_split=None):
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

    model_rfe = feature_selection.RFECV(clone(model), step=1, scoring='accuracy', cv=cv_split)
    model_rfe.fit(X_train, y_train)
    print('do_feat_rfe: Done')
    return X_train.columns.values[model_rfe.get_support()]


# --------------------------------------------------------------------------------------------
def do_boruta(model, X_tr, y_tr):
    selector = BorutaPy(model, n_estimators='auto', verbose=0,
                        random_state=1, max_iter=500)
    selector.fit(X_tr.values, y_tr.values)
    X_filtered = selector.transform(X_tr.values)
    print('do_feat_boruta: Done')
    return pd.DataFrame(X_filtered, columns=X_tr.columns[selector.support_])


# --------------------------------------------------------------------------------------------
def do_pca(df, threshold):
    print('PCA requires feature values to be standardised. Are they?))))')
    pca = PCA().fit_transform(df)
    explained_variance = 0.0
    components = 0

    for var in pca.explained_variance_ratio_:
        explained_variance += var
        components += 1
        if explained_variance >= threshold:
            break
    print('do_feat_pca: Done')
    return PCA(n_components=components).fit_transform(df)


# --------------------------------------------------------------------------------------------
def do_tsne(df, target):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
    tsne_results = tsne_model.fit_transform(df)

    traceTSNE = go.Scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        name=target,
        hoveron=target,
        mode='markers',
        text=target,
        showlegend=True,
        marker=dict(
            size=8,
            color='#c94ff2',
            showscale=False,
            line=dict(
                width=2,
                color='rgb(255, 255, 255)'
            ),
            opacity=0.8
        )
    )
    data = [traceTSNE]

    layout = dict(title='TSNE (T-Distributed Stochastic Neighbour Embedding)',
                  hovermode='closest',
                  yaxis=dict(zeroline=False),
                  xaxis=dict(zeroline=False),
                  showlegend=False,

                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig)
    print('do_feat_tsne: Done')
    return tsne_results


# -------------------------------------------------------------------------------------------
def do_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    X_clustered = kmeans.fit_predict(df)
    trace_Kmeans = go.Scatter(x=df[:, 0], y=df[:, 1],
                              mode="markers",
                              showlegend=False,
                              marker=dict(
                                  size=8,
                                  color=X_clustered,
                                  colorscale='Portland',
                                  showscale=False,
                                  line=dict(
                                      width=2,
                                      color='rgb(255, 255, 255)'
                                  )
                              ))

    layout = go.Layout(
        title='KMeans Clustering',
        hovermode='closest',
        xaxis=dict(
            title='First Principal Component',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Second Principal Component',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=True
    )

    data = [trace_Kmeans]
    fig1 = dict(data=data, layout=layout)
    py.iplot(fig1, filename="svm")
    print('do_feat_kmeans: Done')
    return X_clustered


# --------------------------------------------------------------------------------------------
def do_sel_from_model(X, y, model=None, params=None, n_estimators=1000):
    if params is None:
        raise ValueError('Parameter <params> must be specified.')
    if model is None:
        model = lgb.LGBMClassifier(**params,
                                   n_estimators=n_estimators,
                                   n_jobs=-1)

    selector = SelectFromModel(model, threshold='1.25*median')
    selector.fit(X, y)
    support = selector.get_support()
    features = X.loc[:, support].columns.tolist()
    print(str(len(features)), 'selected features')
    print('do_feat_from_model: Done')
    return features


def do_ks_2samp(train, test):
    rej = []
    not_rej = []

    for col in train.columns:
        statistic, pvalue = ks_2samp(train[col], test[col])
        if pvalue >= statistic:
            not_rej.append(col)
        if pvalue < statistic:
            rej.append(col)

        plt.figure(figsize=(8, 4))
        plt.title("Kolmogorov-Smirnov test for train/test\n"
                  "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, statistic, pvalue))
        sns.kdeplot(train[col], color='blue', shade=True, label='Train')
        sns.kdeplot(test[col], color='green', shade=True, label='Test')

        plt.show()
    return rej, not_rej


def do_ols(train, target_col, fillna=True):
    print('All category cols should be encoded.')
    corr = train.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = train.columns[columns]
    print(selected_columns.shape, ' selected by correlation')

    train = train[selected_columns]
    selected_columns = list(selected_columns)
    selected_columns.remove(target_col)

    if fillna:
        train.fillna(-999, inplace=True)

    def _backwardElimination(x, Y, sl, columns):
        numVars = len(columns)
        for i in tqdm(range(0, numVars)):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if regressor_OLS.pvalues[j].astype(float) == maxVar:
                        x = np.delete(x, j, 1)
                        columns = np.delete(columns, j)

        regressor_OLS.summary()
        return x, columns

    SL = 0.05
    data_modeled, ols_selected_columns = _backwardElimination(train.drop([target_col], 1).values,
                                                              train.target_col.values, SL, selected_columns)

    print(len(ols_selected_columns), ' selected by ols')
    return data_modeled, ols_selected_columns
