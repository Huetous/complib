import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_error


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
def get_hue_oof(model, X, y, X_test,
                cv_scheme='kf', n_splits=5, shuffle=False, seed=0,
                metrics=[('mae', mean_absolute_error)]):
    # -------------------------------------
    # Specify CV scheme
    # -------------------------------------
    if cv_scheme not in ['kf', 'skf', 'ts']:
        raise ValueError('Parameter <cv_scheme> incorrectly specified.')
    else:
        if cv_scheme is 'kf':
            cv_split = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        elif cv_scheme is 'skf':
            cv_split = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        else:
            cv_split = model_selection.TimeSeriesSplit(n_splits=n_splits)

    # -------------------------------------
    # S_train - oof, S_test - pred
    # -------------------------------------
    S_train = np.zeros((X.shape[0],))
    S_test = np.zeros((X_test.shape[0],))
    S_test_tmp = np.empty((n_splits, X_test.shape[0]))

    # -------------------------------------
    # Scores for each fold
    # -------------------------------------
    scores = dict()
    columns = X.columns
    # -------------------------------------
    # Loop for fold
    # -------------------------------------
    for fold_n, (tr_idx, val_idx) in enumerate(cv_split.split(X)):
        if type(X) is np.ndarray:
            X_tr, X_val = X[columns][tr_idx], X[columns][val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
        else:
            X_tr, X_val = X[columns].iloc[tr_idx], X[columns].iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr, X_val, y_val)
        y_pred = model.predict(X_val)

        for (metric_name, metric) in metrics:
            scores[metric_name] = metric(y_val, y_pred)

        S_train[val_idx] = y_pred.ravel()
        S_test_tmp[fold_n, :] = model.predict(X_test)

    print('=' * 60)
    for metric in scores.keys():
        print(f'[{metric}]\t','CV mean:', np.mean(scores[metric]), ', std:', np.std(scores[metric]))
    S_test[:] = S_test_tmp.mean(axis=0)
    return S_train.reshape(-1, 1), S_test.reshape(-1, 1)

# --------------------------------------------------------------------------------------------
