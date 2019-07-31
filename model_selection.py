import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns
from tqdm import tqdm
from sklearn.base import clone
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from huelib.metrics import eval_auc
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------------------------
def show_confusion_matrix(cm, classes, normalize=False, cmap=plt.get_cmap('RdBu')):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# --------------------------------------------------------------------------------------------
def get_hue_oof(params, X, y, X_test,
                cv_scheme='kf', n_splits=5, shuffle=False, seed=42,
                metrics=None, n_estimators=1000,
                verbose=200, early_stopping_rounds=100,
                conf_matrix=False, conf_matrix_norm=False):
    if metrics is None:
        metrics = [('f1', f1_score)]
    if cv_scheme is 'kf':
        cv_split = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    elif cv_scheme is 'skf':
        cv_split = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    else:
        cv_split = model_selection.TimeSeriesSplit(n_splits=n_splits)

    S_train = np.zeros((X.shape[0],))
    S_test = np.zeros((X_test.shape[0],))

    if conf_matrix:
        cms = []
    scores = pd.DataFrame()
    feature_importance = pd.DataFrame()
    for metric_name, metric in metrics:
        scores[metric_name] = np.ndarray((n_splits,))
    columns = X.columns

    for fold_n, (tr_idx, val_idx) in enumerate(cv_split.split(X, y)):
        if type(X) is np.ndarray:
            X_tr, X_val = X[columns][tr_idx], X[columns][val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
        else:
            X_tr, X_val = X[columns].iloc[tr_idx], X[columns].iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params,
                                   n_estimators=n_estimators,
                                   n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  eval_metric=eval_auc,
                  verbose=verbose,
                  early_stopping_rounds=early_stopping_rounds)

        oof_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test,
                                        num_iteration=model.best_iteration_)[:, 1]

        for_metrics = model.predict(X_val)
        for (metric_name, metric) in metrics:
            scores.loc[fold_n, metric_name] = metric(y_val, for_metrics)

        S_train[val_idx] = oof_pred.ravel()
        S_test += test_pred

        if conf_matrix:
            cms.append(confusion_matrix(y_val, for_metrics))

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    S_test /= n_splits

    feature_importance["importance"] /= n_splits
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    plt.figure(figsize=(16, 12))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LGB Features (avg over folds)')

    if conf_matrix:
        cm = np.average(cms, axis=0)
        plt.figure()
        show_confusion_matrix(cm=cm, classes=[0, 1], normalize=conf_matrix_norm)

    print('=' * 60)
    for metric in scores.columns:
        print(f'[{metric}]\t', 'CV mean:', np.mean(scores[metric]), ', std:', np.std(scores[metric]))
    return S_train.reshape(-1, 1), S_test.reshape(-1, 1)


# --------------------------------------------------------------------------------------------

def adversarial_cv(train, test, cols_to_drop=None):
    if cols_to_drop is not None:
        test = test.drop(cols_to_drop, axis=1)
    features = test.columns
    train = train[features]

    train['target'] = 0
    test['target'] = 1

    train_test = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()

    object_columns = list(train_test.select_dtypes('object').columns)

    for f in object_columns:
        lbl = LabelEncoder()
        lbl.fit(list(train_test[f].values))
        train_test[f] = lbl.transform(list(train_test[f].values))

    train, test = model_selection.train_test_split(train_test, test_size=0.33,
                                                   random_state=42, shuffle=True)
    del train_test
    gc.collect()

    train_y = train['target'].values
    test_y = test['target'].values
    del train['target'], test['target']
    gc.collect()

    train = lgb.Dataset(train, label=train_y)
    test = lgb.Dataset(test, label=test_y)

    param = {'num_leaves': 50,
             'min_data_in_leaf': 30,
             'objective': 'binary',
             'max_depth': 5,
             'learning_rate': 0.2,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 44,
             "metric": 'auc',
             "verbosity": -1}

    clf = lgb.train(param, train, 200, valid_sets=[train, test],
                    verbose_eval=50, early_stopping_rounds=50)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), features)),
                               columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False).head(30))
    plt.title('LightGBM Features')
    plt.tight_layout()
