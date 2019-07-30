import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from xgboost import XGBRegressor
from ensemble.sklearnstack import StackingTransformer

import warnings

warnings.filterwarnings("ignore")

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from huelib.ensemble.lowramstack import stack

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor



# from huelib.models import HuetousCatBoost
#
# params = {
#
# }
# cat = HuetousCatBoost(params=params, task='reg', iterations=1000, early_stopping_rounds=100)
# cat.fit(X_train, y_train, X_test, y_test)

# models = [
#     AdaBoostRegressor(),
#      ExtraTreesRegressor()
# ]
#
# S_train, S_test = stack(models, X_train=X_train, y_train=y_train, X_test=X_test,
#                         mode='oof', metric=mean_absolute_error, verbose=2)


# from huelib.models import HuetousCatBoost, HuetousLGB, HuetousXGB
#
# lgb_params = {
#     'objective': 'regression',
#     "metric": 'mean_squared_error',
#     "random_state": 0,
# }
# xgb_params = {
#     'tree_method': 'gpu_hist',
#     'eval_metric': 'mae',
#     "random_state": 0,
# }
# cat_params = {
#     # 'task_type': "GPU",
#     'loss_function': 'RMSE',
#     "random_state": 0,
# }
# hue_lgb = HuetousLGB(lgb_params)
# hue_xgb = HuetousXGB(xgb_params)
# hue_cat = HuetousCatBoost(cat_params)
#
# from huelib.model_selection import get_hue_oof
# from sklearn.metrics import mean_absolute_error

# S_train_lgb, S_test_lgb = get_hue_oof(hue_lgb, X_tr, y_tr, X_test, cv_scheme='kf', n_splits=4, shuffle=False, seed=0,
# metric=mean_absolute_error)
# oof_xgb, preds_xgb = get_hue_oof(hue_xgb, X_tr, y_tr, X_test, cv_scheme='kf', n_splits=4, shuffle=False, seed=0,
#                                  metric=mean_absolute_error)
# oof_cat, preds_cat = get_hue_oof(hue_cat, X_tr, y_tr, X_test, cv_scheme='kf', n_splits=4, shuffle=False, seed=0,
#                                  metric=mean_absolute_error)
# print('lgb\t', mean_absolute_error(y_test, S_test_lgb))
# print('xgb\t', mean_absolute_error(y_test, preds_xgb))
# print('cat\t', mean_absolute_error(y_test, preds_cat))

#
# xgb_params_2 = {
#     "random_state": 0,
#     'num_boost_round': 200,
# }
#

#
# cat_params_2 = {
#     'loss_function': 'RMSE',
#     "random_state": 0,
#     'num_boost_round': 200,
# }
#
# lgb = HueLGB(lgb_params)
# xgb1 = HueXGB(xgb_params_1)
# xgb2 = HueXGB(xgb_params_2)
# cat1 = HueCatBoost(cat_params_1)
# cat2 = HueCatBoost(cat_params_2)
# import time
#
# start = time.time()
# lgb.train(X_train, y_train, X_val, y_val)
# preds = lgb.predict(X_test)
# print('lgb', time.time() - start, ' ', mean_absolute_error(y_test, preds))
#
# start = time.time()
# xgb1.train(X_train, y_train, X_val, y_val)
# preds = xgb1.predict(X_test)
# print('xgb1', time.time() - start, ' ', mean_absolute_error(y_test, preds))
#
# start = time.time()
# xgb2.train(X_train, y_train, X_val, y_val)
# preds = xgb2.predict(X_test)
# print('xgb2', time.time() - start, ' ', mean_absolute_error(y_test, preds))
#
# start = time.time()
# cat1.train(X_train, y_train, X_val, y_val)
# preds = cat1.predict(X_test)
# print('cat1', time.time() - start, ' ', mean_absolute_error(y_test, preds))
#
# start = time.time()
# cat2.train(X_train, y_train, X_val, y_val)
# preds = cat2.predict(X_test)
# print('cat2', time.time() - start, ' ', mean_absolute_error(y_test, preds))
#
# # estimators_L1 = [
# #     ('et', ExtraTreesRegressor(random_state=0, n_jobs=-1,
# #                                n_estimators=100, max_depth=3)),
# #     ('rf', RandomForestRegressor(random_state=0, n_jobs=-1,
# #                                  n_estimators=100, max_depth=3)),
# #
# #     ('xgb', XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
# #                          n_estimators=100, max_depth=3))
# # ]
# #
# # stack = StackingTransformer(estimators=estimators_L1,
# #                             regression=True,
# #                             variant='A',
# #                             metric=mean_absolute_error,
# #                             n_folds=4,
# #                             shuffle=True,
# #                             random_state=0,
# #                             verbose=2)
# # stack.fit(X_train, y_train)
# #
# # S_train = stack.transform(X_train)
# # S_test = stack.transform(X_test)
# #
# # final_estimator = XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
# #                                n_estimators=100, max_depth=3)
# # final_estimator = final_estimator.fit(S_train, y_train)
# # y_pred = final_estimator.predict(S_test)
# # print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))
