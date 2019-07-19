import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from xgboost import XGBRegressor
from huestack.transformer import StackingTransformer

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# get_all_models
# models_eval
#

estimators_L1 = [
    ('et', ExtraTreesRegressor(random_state=0, n_jobs=-1,
                               n_estimators=100, max_depth=3)),
    ('rf', RandomForestRegressor(random_state=0, n_jobs=-1,
                                 n_estimators=100, max_depth=3)),

    ('xgb', XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
                         n_estimators=100, max_depth=3))
]

stack = StackingTransformer(estimators=estimators_L1,
                            regression=True,
                            variant='A',
                            metric=mean_absolute_error,
                            n_folds=4,
                            shuffle=True,
                            random_state=0,
                            verbose=2)
stack.fit(X_train, y_train)

S_train = stack.transform(X_train)
S_test = stack.transform(X_test)

final_estimator = XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
                               n_estimators=100, max_depth=3)
final_estimator = final_estimator.fit(S_train, y_train)
y_pred = final_estimator.predict(S_test)
print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))
