from sklearn import feature_selection, model_selection

from sklearn.base import clone

# RFE
# SelectPercentile
# SelectKBest
# Shuffle Permutation
# --------------------------------------------------------------------------------------------

def do_feat_rfe(model, X_train, y_train, cv_split=None):
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)

    model_before_copy = clone(model)
    base_res = model_selection.cross_validate(model_before_copy, X_train, y_train, cv=cv_split)
    print('Before RFE Training Shape Old: ', X_train.shape)
    print('Before RFE Training Columns Old: ', X_train.columns.values)
    print('Before Training with bin score mean: {:.3f}'.format(base_res['train_score'].mean() * 100))
    print('Before Test with bin score mean: {:.3f}'.format(base_res['test_score'].mean() * 100))
    print('Before Test with bin score 3*std: +/- {:.3f}'.format(base_res['test_score'].std() * 3 * 100))
    print('-' * 15)
    del model_before_copy

    model_copy_for_rfe = clone(model)
    model_rfe = feature_selection.RFECV(model_copy_for_rfe, step=1, scoring='accuracy', cv=cv_split)
    model_rfe.fit(X_train, y_train)
    X_rfe = X_train.columns.values[model_rfe.get_support()]
    del model_copy_for_rfe

    model_copy_after_rfe = clone(model)
    rfe_res = model_selection.cross_validate(model_copy_after_rfe, X_train[X_rfe], y_train, cv=cv_split)
    print('After RFE Training Shape New: ', X_train[X_rfe].shape)
    print('After RFE Training Columns New: ', X_rfe)
    print('After Training with bin score mean: {:.3f}'.format(rfe_res['train_score'].mean() * 100))
    print('After Test with bin score mean: {:.3f}'.format(rfe_res['test_score'].mean() * 100))
    print('After Test with bin score 3*std: +/- {:.3f}'.format(rfe_res['test_score'].std() * 3 * 100))
    print('-' * 15)

# --------------------------------------------------------------------------------------------
# def do_drop_highly_corr_feats():
#     corr_matrix = X.corr().abs()
#
#     # Select upper triangle of correlation matrix
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#
#     # Find index of feature columns with correlation greater than 0.95
#     to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
#     X = X.drop(to_drop, axis=1)
#     X_test = X_test.drop(to_drop, axis=1)
#     result_dict_lgb_lgb = artgor_utils.train_model_regression(X, X_test, y, params=params,
#                                                               folds=folds, model_type='lgb',
#