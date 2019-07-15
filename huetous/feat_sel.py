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

plotly.tools.set_credentials_file(username='daddudota3', api_key='PjqulG0oXHlrVgWexu2q')


# --------------------------------------------------------------------------------------------
def do_feat_rfe(model, X_train, y_train, cv_split=None):
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

    model_rfe = feature_selection.RFECV(clone(model), step=1, scoring='accuracy', cv=cv_split)
    model_rfe.fit(X_train, y_train)
    print('do_feat_rfe: Done')
    return X_train.columns.values[model_rfe.get_support()]


# --------------------------------------------------------------------------------------------
def do_feat_boruta(model, X_tr, y_tr):
    selector = BorutaPy(model, n_estimators='auto', verbose=0,
                        random_state=1, max_iter=500)
    selector.fit(X_tr.values, y_tr.values)
    X_filtered = selector.transform(X_tr.values)
    print('do_feat_boruta: Done')
    return pd.DataFrame(X_filtered, columns=X_tr.columns[selector.support_])


# --------------------------------------------------------------------------------------------
def do_feat_pca(df, threshold):
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
def do_feat_tsne(df, target):
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
def do_feat_kmeans(df, n_clusters):
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


