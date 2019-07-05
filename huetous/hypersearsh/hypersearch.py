# --------------------------------------------
# HYPER PARAMETERS SEARCH
# --------------------------------------------
from sklearn import model_selection
import time
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# Bayesian optimization
# Grid Search
# Random Search
# --------------------------------------------------------------------------------------------
def get_param_grid(models_names, regression=True):
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]
    print('Запрос сетки параметров...')

    param_grid = []
    if regression:
        params = {
            'AdaBoostRegressor': [{'n_estimators': grid_n_estimator,
                                   'learning_rate': grid_learn,
                                   'random_state': grid_seed}],
            # Extra Trees
            'ExtraTreesRegressor': [{'n_estimators': grid_n_estimator,
                                     'criterion': grid_criterion,
                                     'max_depth': grid_max_depth,
                                     'random_state': grid_seed
                                     }],
            # Gradient Boosting
            'GradientBoostingRegressor': [{'learning_rate': grid_learn,
                                           'n_estimators': grid_n_estimator,
                                           'max_depth': grid_max_depth,
                                           'random_state': grid_seed}],
            # Random Forest
            'RandomForestRegressor': [{'n_estimators': grid_n_estimator,
                                       'max_depth': grid_max_depth,
                                       'oob_score': [True],
                                       'random_state': grid_seed}],
            # Gaussian Process
            'GaussianProcessRegressor': [{'random_state': grid_seed}],

            # Passive Aggressive
            'PassiveAggressiveRegressor': [{}],

            # SGDRegressor
            'SGDRegressor': [{'C': [1, 2, 3, 4, 5]}],

            # KNN
            'KNeighborsRegressor': [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
                                     'weights': ['uniform', 'distance'],
                                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}],
            # SVR
            'SVR': [{'C': [1, 2, 3, 4, 5],
                     'gamma': grid_ratio,
                     'decision_function_shape': ['ovo', 'ovr'],
                     'probability': [False],
                     'random_state': grid_seed}],

            # NuSVR
            'NuSVR': [{'C': [1, 2, 3, 4, 5]}],

            # Linear SVR
            'LinearSVR': [{'C': [1, 2, 3, 4, 5]}],

            # Decision Tree
            'DecisionTreeRegressor': [{'max_depth': grid_max_depth,
                                       'splitter': ['best', 'random']}],

            # Extra Tree
            'ExtraTreeRegressor': [{'max_depth': grid_max_depth,
                                    'splitter': ['best', 'random']}],

            # XGB
            'XGBRegressor': [{'learning_rate': grid_learn,
                              'max_depth': [1, 2, 4, 6, 8, 10],
                              'n_estimators': grid_n_estimator,
                              'seed': grid_seed}]}
    else:
        params = {
            # AdaBoost
            'AdaBoostClassifier': [{'n_estimators': grid_n_estimator,
                                    'learning_rate': grid_learn,
                                    'random_state': grid_seed}],

            # Bagging
            'BaggingClassifier': [{'n_estimators': grid_n_estimator,
                                   'max_samples': grid_ratio,
                                   'random_state': grid_seed}],

            # Extra Trees
            'ExtraTreesClassifier': [{'n_estimators': grid_n_estimator,
                                      'criterion': grid_criterion,
                                      'max_depth': grid_max_depth,
                                      'random_state': grid_seed
                                      }],

            # Gradient Boosting
            'GradientBoostingClassifier': [{'learning_rate': grid_learn,
                                            'n_estimators': grid_n_estimator,
                                            'max_depth': grid_max_depth,
                                            'random_state': grid_seed}],

            # Random Forest
            'RandomForestClassifier': [{'n_estimators': grid_n_estimator,
                                        'criterion': grid_criterion,
                                        'max_depth': grid_max_depth,
                                        'oob_score': grid_bool,
                                        'random_state': grid_seed}],

            # Gaussian Process
            'GaussianProcessClassifier': [{'max_iter_predict': grid_n_estimator,
                                           'random_state': grid_seed}],

            # Logistic
            'LogisticRegressionCV': [{'fit_intercept': grid_bool,
                                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', ],
                                      'random_state': grid_seed}],

            # Passive Aggressive
            'PassiveAggressiveClassifier': [{'C': [1, 2, 3, 4, 5],
                                             'random_state': grid_seed,
                                             'early_stopping': grid_bool}],

            # Ridge
            'RidgeClassifierCV': [{}],

            # SGD
            'SGDClassifier': [{}],

            # Perceptron
            'Perceptron': [{}],

            # Bernoulli
            'BernoulliNB': [{'alpha': grid_ratio}],

            # Gaussian
            'GaussianNB': [{}],

            # KNN
            'KNeighborsClassifier': [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
                                      'weights': ['uniform', 'distance'],
                                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}],

            # SVC
            'SVC': [{'C': [1, 2, 3, 4, 5],
                     'gamma': grid_ratio,
                     'decision_function_shape': ['ovo', 'ovr'],
                     'probability': [True],
                     'random_state': grid_seed}],

            # NuSVC
            'NuSVC': [{}],

            # Linear SVC
            'LinearSVC': [{}],

            # Decision Tree
            'DecisionTreeClassifier': [{'max_depth': grid_max_depth,
                                        'splitter': ['best', 'random']}],

            # Extra Tee
            'ExtraTreeClassifier': [{'max_depth': grid_max_depth,
                                     'splitter': ['best', 'random']}],

            # Linear Discriminant
            'LinearDiscriminantAnalysis': [{}],

            # Quadratic Discriminant
            'QuadraticDiscriminantAnalysis': [{}],

            # XGB
            'XGBClassifier': [{'learning_rate': grid_learn,
                               'max_depth': [1, 2, 4, 6, 8, 10],
                               'n_estimators': grid_n_estimator,
                               'seed': grid_seed}]}

        for name in models_names:
            param_grid.append(params[name])
        print('Объем возвращаемой сетки параметров:', len(param_grid))

    return param_grid


# --------------------------------------------------------------------------------------------
def do_param_grid_search(models, grid_param, X, y, cv_split=None):
    if cv_split is None:
        cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    start_total = time.perf_counter()
    for clf, param in zip(models, grid_param):
        start = time.perf_counter()
        best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
        best_search.fit(X, y)
        run = time.perf_counter() - start
        best_param = best_search.best_params_
        print('Best params for {} is {}, runtime: {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param)

    run_total = time.perf_counter() - start_total
    print('Total optimization time: {:.2f} minutes.'.format(run_total / 60))
    print('-' * 15)


# --------------------------------------------------------------------------------------------
