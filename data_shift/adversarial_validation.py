import pandas as pd
import gc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import model_selection
import seaborn as sns


def get_adv_val(X, X_test):
    X_test.drop(["target"], 1, inplace=True)

    train, val = X[X.probas < 0.9], X[X.probas >= 0.9]
    weights = X['probas']

    train = train.drop(["probas"], 1)
    val = val.drop(["probas"], 1)

    X_train, y_train = train.drop("target", 1), train.target
    X_val, y_val = val.drop("target", 1), val.target

    print("Train shape: {}\nValidation shape: {}\nTest shape: {}".format(X_train.shape, X_val.shape, X_test.shape))
    return X_train, y_train, X_val, y_val, X_test, weights


def get_adv_feats_by_one(X_train, Y_train, X_test):
    X_train['target'] = Y_train
    X_test['target'] = 0

    X_train["is_test"] = 0
    X_test["is_test"] = 1
    assert (np.all(X_train.columns == X_test.columns))

    total = pd.concat([X_train, X_test])
    X_split = total.drop(["is_test", "target"], axis=1)
    y_split = total.is_test

    model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5)
    feats = []
    scores = []
    for col in X_split.columns:
        score = cross_val_score(model, pd.DataFrame(X_split[col]), y_split, cv=2, scoring='roc_auc')
        feats.append(col)
        scores.append(np.mean(score))
        print(col, np.mean(score))

    df = pd.DataFrame()
    df['feats'] = feats
    df['scores'] = scores
    return df


def get_adv_train(X_train, Y_train, X_test, seed=42):
    X_train['target'] = Y_train
    X_test['target'] = 0

    X_train["is_test"] = 0
    X_test["is_test"] = 1
    assert (np.all(X_train.columns == X_test.columns))

    total = pd.concat([X_train, X_test])

    X_split = total.drop(["is_test", "target"], axis=1)
    y_split = total.is_test

    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_split, y_split, test_size=0.2,
                                                                          random_state=seed)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_valid, label=y_valid)

    params = {
        'max_depth': 9,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'tree_method': 'gpu_hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': seed,
    }
    print('=' * 40)
    print('Train')
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    clf = xgb.train(dtrain=dtrain,
                    num_boost_round=300, evals=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=100, params=params)
    print('=' * 40)
    feature_imp = pd.DataFrame(sorted(zip(clf.get_score().keys(), clf.get_score().values())),
                               columns=['Feature', 'Value'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()

    print('Extract')
    X_train = total[total.is_test == 0]
    X_test = total[total.is_test == 1]

    X_train.drop(['is_test'], 1, inplace=True)
    X_test.drop(['is_test'], 1, inplace=True)
    dval = xgb.DMatrix(data=X_train.drop(['target'], 1))
    X_train['probas'] = clf.predict(dval)
    X_train.sort_values(["probas"], ascending=False, inplace=True)
    return X_train, X_test, feature_imp
