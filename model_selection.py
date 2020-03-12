import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix
from preprocessing import DoubleValidationEncoder


# --------------------------------------------------------------------------------------------
def show_confusion_matrix(cm, classes, cmap=plt.get_cmap('RdBu')):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# --------------------------------------------------------------------------------------------
# Return out-of-fold predictions, with summary of model performance on each iteration
def get_oof(clf, X, y, X_test,
                 splits=None,
                 metrics=None, conf_matrix=False,
                 encoder=None, encode_cols=None,
                 train_cols=None):
    if splits is None:
        raise ValueError('Parameter <splits> must be specified')

    S_train = np.zeros(len(X))
    S_test = np.zeros(len(X_test))
    n_splits = len(splits)

    if conf_matrix:
        cms = []
    feature_importance = pd.DataFrame()

    if metrics:
        scores = pd.DataFrame()
        for name, metric in metrics:
            scores[name] = np.ndarray((n_splits,))

    if train_cols is None and type(X) is pd.core.frame.DataFrame:
        train_cols = list(X.columns)

    for fold_n, (tr, val) in enumerate(splits):
        print(f'Fold #{fold_n}')
        if type(X) is pd.core.frame.DataFrame:
            X_tr, X_val = X[train_cols].iloc[tr], X[train_cols].iloc[val]
            y_tr, y_val = y[tr], y[val]
        else:
            X_tr, X_val = X[tr], X[val]
            y_tr, y_val = y[tr], y[val]

        X_t = X_test.copy()
        if encoder:
            if encode_cols is None:
                encode_cols = list(X.columns)
            enc = DoubleValidationEncoder(cols=encode_cols, encoder=encoder, splits=splits)
            X_tr = enc.fit_transform(X_tr, y_tr)
            X_val = enc.transform(X_val)
            X_t = enc.transform(X_t)

        clf.fit(X_tr, y_tr, X_val, y_val)

        oof_pred = clf.predict(X_val)
        test_pred = clf.predict(X_t)

        if metrics:
            for name, metric in metrics:
                scores.loc[fold_n, name] = metric(y_val, oof_pred)

        S_train[val] = oof_pred.ravel()
        S_test += test_pred

        if conf_matrix:
            cms.append(confusion_matrix(y_val, oof_pred))

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = train_cols

        if hasattr(clf, 'feature_importances_'):
            fold_importance["importance"] = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            fold_importance["importance"] = clf.coef_[0]
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        else:
            print('Model doesn`t support feature importance.')

    S_test /= n_splits

    feature_importance["importance"] /= n_splits
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    plt.figure(figsize=(16, 12))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance", ascending=False))

    if conf_matrix:
        cm = np.average(cms, axis=0)
        plt.figure()
        show_confusion_matrix(cm=cm, classes=[0, 1])

    print('=' * 60)
    if metrics is not None:
        for metric in scores.columns:
            print(f'[{metric}]\t', 'CV mean:', np.mean(scores[metric]), ', std:', np.std(scores[metric]))
    return S_train.reshape(-1, 1), S_test.reshape(-1, 1).ravel()
