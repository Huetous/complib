from bayes_opt import BayesianOptimization
import lightgbm as lgb


# --------------------------------------------------------------------------------------------
# Performs Bayesian optimisation of parameters
def do_bayes(X, y, init_round=5, opt_round=15,
             n_folds=5, random_seed=42, n_estimators=100,
             learning_rate=0.05):
    dtrain = lgb.Dataset(data=X, label=y,
                         #categorical_feature = categorical_feats,
                         free_raw_data=False)

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain,
                 min_child_samples, min_child_weight, subsample, subsample_freq, colsample_bytree):
        params = {'objective': 'binary',
                  'num_iterations': n_estimators,
                  'learning_rate': learning_rate,
                  'early_stopping_round': 100,
                  'metric': 'auc',
                  "num_leaves": int(round(num_leaves)),
                  'feature_fraction': max(min(feature_fraction, 1), 0),
                  'bagging_fraction': max(min(bagging_fraction, 1), 0),
                  'max_depth': int(round(max_depth)),
                  'lambda_l1': max(lambda_l1, 0),
                  'lambda_l2': max(lambda_l2, 0),
                  'min_split_gain': min_split_gain,
                  'min_child_samples': int(min_child_samples),
                  'min_child_weight': min_child_weight,
                  'subsample': subsample,
                  'subsample_freq': int(subsample_freq),
                  'colsample_bytree': colsample_bytree}

        cv_result = lgb.cv(params, dtrain,
                           nfold=n_folds, seed=random_seed,
                           stratified=True, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (45, 55),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 9),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_samples': (65, 75),
                                            'min_child_weight': (5, 50),
                                            'subsample': (0.7, 0.9),
                                            'subsample_freq': (2, 4),
                                            'colsample_bytree': (0.7, 0.9)
                                            }, random_state=0)

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    max_ = 0
    for i in range(len(lgbBO.res)):
        if lgbBO.res[i]['target'] > max_:
            max_ = i

    return lgbBO.res[max_]['params']

