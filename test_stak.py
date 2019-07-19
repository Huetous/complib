from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
from huetous.ensemble import ensemble as ens
from huetous import eda

boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [
    ('adar', ensemble.AdaBoostRegressor()),
    ('baggr', ensemble.BaggingRegressor()),
    ('etsr', ensemble.ExtraTreesRegressor()),
    ('gbr', ensemble.GradientBoostingRegressor()),
    ('rfr', ensemble.RandomForestRegressor()),

    ('gpr', gaussian_process.GaussianProcessRegressor()),

    ('lr', linear_model.LogisticRegressionCV()),
    ('par', linear_model.PassiveAggressiveRegressor()),
    ('sgdr', linear_model.SGDRegressor()),

    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),

    ('knnr', neighbors.KNeighborsRegressor()),

    ('svr', svm.SVR()),
    ('nusvr', svm.NuSVR()),
    ('lsvr', svm.LinearSVR()),

    ('dtr', tree.DecisionTreeRegressor()),
    ('etr', tree.ExtraTreeRegressor()),

    ('lda', discriminant_analysis.LinearDiscriminantAnalysis()),
    ('qda', discriminant_analysis.QuadraticDiscriminantAnalysis()),

    ('xgbr', XGBRegressor())
]

cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
mla_compare, mla_predict = ens.models_eval(models, X,  y, cv_split)
eda.correlation_heatmap(mla_predict)
#
# # grid_n_estimators = [10, 50, 100, 300]
# # grid_ratio = [.1, .25, .5, .75, 1.0]
# # grid_learn = [.01, .03, .05, .1, .25]
# # grid_max_depth = [2, 4, 6, 8, 10, None]
# # grid_min_samples = [5, 10, .03, .05, .10]
# # grid_criterion = ['gini', 'entropy']
# # grid_bool = [True, False]
# # grid_seed = [0]
# #
# # grid_param = [
# #     # AdaBoost
# #     ('adar',
# #     [{'n_estimators': grid_n_estimators,
# #       'learning_rate': grid_learn,
# #       'random_state': grid_seed}]),
# #     # Bagging
# #     ('baggr',
# #     [{'n_estimators': grid_n_estimators,
# #       'max_samples': grid_ratio,
# #       'random_state': grid_seed}]),
# #     # ExtraTrees
# #     ('etsr',
# #     [{'n_estimators': grid_n_estimators,
# #       'criterion': grid_criterion,
# #       'max_depth': grid_max_depth,
# #       'random_state': grid_seed
# #       }]),
# #     # GradientBoosting
# #     ('gbr',
# #     [{'learning_rate': [.05],
# #       'n_estimators': [300],
# #       'max_depth': grid_max_depth,
# #       'random_state': grid_seed}]),
# #     # RandomForest
# #     ('rfr',
# #     [{'n_estimators': grid_n_estimators,
# #       'criterion': grid_criterion,
# #       'max_depth': grid_max_depth,
# #       'oob_score': [True],
# #       'random_state': grid_seed}]),
# #
# #     # GaussianProcess
# #     ('gpr',
# #     [{'max_iter_predict': grid_n_estimators,
# #       'random_state': grid_seed}]),
# #
# #     # LogisticRegression
# #     ('lr',
# #     [{'fit_intercept': grid_bool,
# #       'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', ],
# #       'random_state': grid_seed}]),
# #     # Bernouli
# #     [{'alpha': grid_ratio}],
# #     # Gaussian
# #     [{}],
# #     # KNN
# #     [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
# #       'weights': ['uniform', 'distance'],
# #       'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}],
# #     # SVC
# #     [{'C': [1, 2, 3, 4, 5],
# #       'gamma': grid_ratio,
# #       'decision_function_shape': ['ovo', 'ovr'],
# #       'probability': [True],
# #       'random_state': grid_seed}],
# #     # XGB
# #     [{'learning_rate': grid_learn,
# #       'max_depth': [1, 2, 4, 6, 8, 10],
# #       'n_estimators': grid_n_estimators,
# #       'seed': grid_seed}]
# # ]
#
# S_train, S_test = stack(models,
#                         X_train, y_train, X_test,
#                         regression=True,
#                         mode='oof_pred_bag',
#                         save_dir=None,
#                         metric=mean_absolute_error,
#                         n_folds=4,
#                         shuffle=True,
#                         random_state=0,
#                         verbose=2)
#
# model = XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
#                      n_estimators=100, max_depth=3)
#
# # Fit 2nd level model
# model = model.fit(S_train, y_train)
#
# # Predict
# y_pred = model.predict(S_test)
#
# # Final prediction score
# print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))
