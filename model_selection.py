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


# --------------------------------------------------------------------------------------------
def get_best_models_by_corr(predictions, target_col='Target', threshhold=.70):
    target = predictions[target_col]
    data = predictions.drop([target_col], axis=1)

    print('Поиск лучших моделей...')
    corr_with_target = pd.DataFrame()
    index = 0
    corr_sum = 0
    for col in data:
        if np.abs(np.corrcoef(data[col], target)[0, 1]) > threshhold:
            corr_with_target.loc[index, 'Model Name'] = col
            corr_with_target.loc[index, 'Correlation with target'] = np.corrcoef(data[col], target)[0, 1]
        index += 1

    for model_name in corr_with_target['Model Name']:
        for other_model_name in corr_with_target['Model Name']:
            if model_name == other_model_name:
                continue
            else:
                corr_sum += np.abs(np.corrcoef(data[model_name], data[other_model_name])[0, 1])
        index = corr_with_target[corr_with_target['Model Name'] == model_name].index[0]
        corr_with_target.loc[index, 'Corr. sum with other models'] = corr_sum
        corr_sum = 0
    corr_with_target.sort_values(['Correlation with target', 'Corr. sum with other models'],
                                 ascending=[False, True], inplace=True)

    print(corr_with_target)
    print('Количество запрошенных моделей: ', corr_with_target.shape[0], '\n')
    return corr_with_target['Model Name'].tolist()


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
