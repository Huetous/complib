import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns
from tqdm import tqdm


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
def confusion_matrix(cm, classes,normalize=False, cmap=plt.get_cmap('RdBu')):
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
def cross_validate(models, X_tr, y_tr, X_val, y_te,
                   cv_scheme='kf', n_splits=3, shuffle=False, seed=0,
                   plot=False):
    if cv_scheme not in ['kf', 'skf']:
        raise ValueError('Parameter <cv_scheme> incorrectly specified.')
    else:
        if cv_scheme is 'kf':
            cv_split = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        else:
            cv_split = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    cols = ['Model', 'Train Acc Mean', 'Train Acc 3*std', 'Test Acc Mean', 'Test Acc 3*std', 'Fit Time']
    compare = pd.DataFrame(columns=cols)
    preds = pd.DataFrame()
    preds['Target'] = y_te

    row_index = 0
    for (name, model) in tqdm(models):
        model_name = model.__class__.__name__
        compare.loc[row_index, 'Model'] = model_name

        cv_res = model_selection.cross_validate(model, X_tr, y_tr, cv=cv_split)

        compare.loc[row_index, 'Train Acc Mean'] = cv_res['train_score'].mean()
        compare.loc[row_index, 'Train Acc 3*std'] = cv_res['train_score'].std() * 3

        compare.loc[row_index, 'Test Acc Mean'] = cv_res['test_score'].mean()
        compare.loc[row_index, 'Test Acc 3*std'] = cv_res['test_score'].std() * 3

        compare.loc[row_index, 'Fit Time'] = cv_res['fit_time'].mean()

        model.fit(X_tr, y_tr)
        preds[model_name] = model.predict(X_val)
        row_index += 1
    compare.sort_values(by=['Test Acc Mean'], ascending=False, inplace=True)

    if plot:
        _, ax = plt.subplots(figsize=(12, 6))
        _ = sns.barplot(x='Test Acc Mean', y='Model', data=compare)
        plt.show()
    return compare, preds
# --------------------------------------------------------------------------------------------
