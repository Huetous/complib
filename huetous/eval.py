import numpy as np
from sklearn import model_selection
from sklearn import feature_selection
from sklearn.base import clone
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import time
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import seaborn as sns


# def kfold(train, targets, model, num_epochs, k=5):
#     num_val_samples = len(train) // k
#     all_scores = []
#     all_mae_histories = []
#     for i in range(k):
#         print("Processing fold №", i)
#         val_data = train[i * num_val_samples: (i + 1) * num_val_samples]
#         val_targets = targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#         part_train = np.concatenate(
#             [train[: i * num_val_samples],
#              train[(i + 1) * num_val_samples:]], axis=0
#         )
#         part_targets = np.concatenate(
#             [targets[: i * num_val_samples],
#              targets[(i + 1) * num_val_samples:]], axis=0
#         )
#         # model.fit(part_train, part_targets, epochs=num_epochs, batch_size=1, verbose=0)
#         # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#         # all_scores.append(val_mae)
#         history = model.fit(part_train, part_targets,
#                             validation_data=(val_data, val_targets),
#                             epochs=num_epochs,
#                             batch_size=1, verbose=0)
#         mae_history = history.history['val_mean_absolute_score']
#         all_mae_histories.append(mae_history)
#       avg_mae_hist = [np.mean([x[i] for x in all_mae_history for i in range(num_epochs)])

# CV
# --------------------------------------------------------------------------------------------
def get_best_models_by_corr(predictions, target_col='Target', threshhold=.70):
    target = predictions[target_col]
    data = predictions.drop([target_col], axis=1)

    print('Поиск лучших моделей...')
    corr_with_target = pd.DataFrame()
    index = 0
    corr_sum = 0
    for col in data:
        if np.corrcoef(data[col], target)[0, 1] > threshhold:
            corr_with_target.loc[index, 'Model Name'] = col
            corr_with_target.loc[index, 'Correlation with target'] = np.corrcoef(data[col], target)[0, 1]
        index += 1

    for model_name in corr_with_target['Model Name']:
        for other_model_name in corr_with_target['Model Name']:
            if model_name == other_model_name:
                continue
            else:
                corr_sum += np.corrcoef(data[model_name], data[other_model_name])[0, 1]
        index = corr_with_target[corr_with_target['Model Name'] == model_name].index[0]
        corr_with_target.loc[index, 'Corr. sum with other models'] = corr_sum
        corr_sum = 0
    corr_with_target.sort_values(['Correlation with target', 'Corr. sum with other models'],
                                 ascending=[False, True], inplace=True)

    print(corr_with_target)
    print('Количество запрошенных моделей: ', corr_with_target.shape[0],'\n')
    return corr_with_target['Model Name'].tolist()
# --------------------------------------------------------------------------------------------
def show_confusion_matrix(cm, classes,
                     normalize=False,
                     title='Confusion matrix',
                     cmap=plt.get_cmap('RdBu')):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
def do_models_eval(models, X, y, cv_split=None, graph=False):
    MLA_cols = ['MLA Name', 'MLA Params', 'MLA Train Acc Mean', 'MLA Test Acc Mean', 'MLA Test Acc 3*std',
                'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_cols)
    MLA_predict = pd.DataFrame()
    MLA_predict['Target'] = y

    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    row_index = 0
    for (name, alg) in models:
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Params'] = str(alg.get_params())

        cv_res = model_selection.cross_validate(alg, X, y, cv=cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_res['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Acc Mean'] = cv_res['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Acc Mean'] = cv_res['test_score'].mean()

        MLA_compare.loc[row_index, 'MLA Test Acc 3*std'] = cv_res['test_score'].std() * 3

        alg.fit(X, y)
        MLA_predict[MLA_name] = alg.predict(X)
        row_index += 1

    MLA_compare.sort_values(by=['MLA Test Acc Mean'], ascending=False, inplace=True)

    if graph:
        _, ax = plt.subplots(figsize=(12, 6))
        _ = sns.barplot(x='MLA Test Acc Mean', y='MLA Name', data=MLA_compare)
        plt.show()
    return MLA_compare, MLA_predict
# --------------------------------------------------------------------------------------------
