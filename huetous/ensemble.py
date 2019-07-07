from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier, XGBRegressor
from sklearn import model_selection
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
import pandas as pd
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------------------------
def get_models(regression=True, names=None):
    print('Запрос моделей...')
    models_reg = {
        'AdaBoostRegressor': ensemble.AdaBoostRegressor(),
        'BaggingRegressor': ensemble.BaggingRegressor(),
        'ExtraTreesRegressor': ensemble.ExtraTreesRegressor(),
        'GradientBoostingRegressor': ensemble.GradientBoostingRegressor(),
        'RandomForestRegressor': ensemble.RandomForestRegressor(),

        'GaussianProcessRegressor': gaussian_process.GaussianProcessRegressor(),

        'LogisticRegression': linear_model.LogisticRegression(),
        'PassiveAggressiveRegressor': linear_model.PassiveAggressiveRegressor(),
        'SGDRegressor': linear_model.SGDRegressor(),

        'KNeighborsRegressor': neighbors.KNeighborsRegressor(),

        'SVR': svm.SVR(),
        'NuSVR': svm.NuSVR(),
        'LinearSVR': svm.LinearSVR(),

        'DecisionTreeRegressor': tree.DecisionTreeRegressor(),
        'ExtraTreeRegressor': tree.ExtraTreeRegressor(),

        'XGBRegressor': XGBRegressor()}
    models_clf = {
        'AdaBoostClassifier': ensemble.AdaBoostClassifier(),
        'BaggingClassifier': ensemble.BaggingClassifier(),
        'ExtraTreesClassifier': ensemble.ExtraTreesClassifier(),
        'GradientBoostingClassifier': ensemble.GradientBoostingClassifier(),
        'RandomForestClassifier': ensemble.RandomForestClassifier(),

        'GaussianProcessClassifier': gaussian_process.GaussianProcessClassifier(),

        'LogisticRegressionCV': linear_model.LogisticRegressionCV(),
        'PassiveAggressiveClassifier': linear_model.PassiveAggressiveClassifier(),
        'RidgeClassifierCV': linear_model.RidgeClassifierCV(),
        'SGDClassifier': linear_model.SGDClassifier(),
        'Perceptron': linear_model.Perceptron(),

        'BernoulliNB': naive_bayes.BernoulliNB(),
        'GaussianNB': naive_bayes.GaussianNB(),

        'KNeighborsClassifier': neighbors.KNeighborsClassifier(),

        'SVC': svm.SVC(probability=True),
        'NuSVC': svm.NuSVC(probability=True),
        'LinearSVC': svm.LinearSVC(),

        'DecisionTreeClassifier': tree.DecisionTreeClassifier(),
        'ExtraTreeClassifier': tree.ExtraTreeClassifier(),

        'LinearDiscriminantAnalysis': discriminant_analysis.LinearDiscriminantAnalysis(),
        'QuadraticDiscriminantAnalysis': discriminant_analysis.QuadraticDiscriminantAnalysis(),

        'XGBClassifier': XGBClassifier()}

    return_models = []
    if names is None:
        if regression:
            for model_name in models_reg:
                return_models.append((model_name, models_reg[model_name]))
            print('Количество возвращенных моделей: ', len(return_models), '\n')
            return return_models
        else:
            for model_name in models_clf:
                return_models.append((model_name, models_clf[model_name]))
            print('Количество возвращенных моделей: ', len(return_models), '\n')
            return return_models
    else:
        if regression:
            for name in names:
                return_models.append((name, models_reg[name]))
        else:
            for name in names:
                return_models.append((name, models_clf[name]))
        # if names is not None:
        #     if regression:
        #         if 'AdaBoostRegressor' in names:
        #             models.append(('AdaBoost', ensemble.AdaBoostRegressor()))
        #         if 'BaggingRegressor' in names:
        #             models.append(('Bagging', ensemble.BaggingRegressor()))
        #         if 'ExtraTreesRegressor' in names:
        #             models.append(('ExtraTrees', ensemble.ExtraTreesRegressor()))
        #         if 'GradientBoostingRegressor' in names:
        #             models.append(('GradientBoosting', ensemble.GradientBoostingRegressor()))
        #         if 'RandomForestRegressor' in names:
        #             models.append(('RandomForest', ensemble.RandomForestRegressor()))
        #
        #         if 'GaussianProcessRegressor' in names:
        #             models.append(('GaussianProcess', gaussian_process.GaussianProcessRegressor()))
        #
        #         if 'PassiveAggressiveRegressor' in names:
        #             models.append(('PassiveAggressive', linear_model.PassiveAggressiveRegressor()))
        #         if 'SGDRegressor' in names:
        #             models.append(('SGD', linear_model.SGDRegressor()))
        #
        #         if 'KNeighborsRegressor' in names:
        #             models.append(('KNeighbors', neighbors.KNeighborsRegressor()))
        #
        #         if 'SVR' in names:
        #             models.append(('SVR', svm.SVR()))
        #         if 'NuSVR' in names:
        #             models.append(('NuSVR', svm.NuSVR()))
        #         if 'LinearSVR' in names:
        #             models.append(('LinearSVR', svm.LinearSVR()))
        #
        #         if 'DecisionTreeRegressor' in names:
        #             models.append(('DecisionTree', tree.DecisionTreeRegressor()))
        #         if 'ExtraTreeRegressor' in names:
        #             models.append(('ExtraTree', tree.ExtraTreeRegressor()))
        #
        #         if 'XGBRegressor' in names:
        #             models.append(('XGB', XGBRegressor()))
        #     else:
        #         if 'AdaBoostClassifier' in names:
        #             models.append(('AdaBoost', ensemble.AdaBoostClassifier()))
        #         if 'ExtraTreesClassifier' in names:
        #             models.append(('ExtraTrees', ensemble.ExtraTreesClassifier()))
        #         if 'BaggingClassifier' in names:
        #             models.append(('Bagging', ensemble.BaggingClassifier()))
        #         if 'GradientBoostingClassifier' in names:
        #             models.append(('GradientBoosting', ensemble.GradientBoostingClassifier()))
        #         if 'RandomForestClassifier' in names:
        #             models.append(('RandomForest', ensemble.RandomForestClassifier()))
        #
        #         if 'GaussianProcessClassifier' in names:
        #             models.append(('GaussianProcess', gaussian_process.GaussianProcessClassifier()))
        #
        #         if 'LogisticRegressionCV' in names:
        #             models.append(('LogisticRegression', linear_model.LogisticRegressionCV()))
        #         if 'PassiveAggressiveClassifier' in names:
        #             models.append(('PassiveAggressive', linear_model.PassiveAggressiveClassifier()))
        #         if 'RidgeClassifierCV' in names:
        #             models.append(('RidgeCV', linear_model.RidgeClassifierCV()))
        #         if 'SGDClassifier' in names:
        #             models.append(('SGD', linear_model.SGDClassifier()))
        #         if 'Perceptron' in names:
        #             models.append(('Perceptron', linear_model.Perceptron()))
        #
        #         if 'BernoulliNB' in names:
        #             models.append(('BernoulliNB', naive_bayes.BernoulliNB()))
        #         if 'GaussianNB' in names:
        #             models.append(('GaussianNB', naive_bayes.GaussianNB()))
        #
        #         if 'KNeighborsClassifier' in names:
        #             models.append(('KNeighbors', neighbors.KNeighborsClassifier()))
        #
        #         if 'SVC' in names:
        #             models.append(('SVC', svm.SVC(probability=True)))
        #         if 'NuSVC' in names:
        #             models.append(('NuSVC', svm.NuSVC(probability=True)))
        #         if 'LinearSVC' in names:
        #             models.append(('LinearSVC', svm.LinearSVC()))
        #
        #         if 'DecisionTreeClassifier' in names:
        #             models.append(('DecisionTree', tree.DecisionTreeClassifier()))
        #         if 'ExtraTreeClassifier' in names:
        #             models.append(('ExtraTree', tree.ExtraTreeClassifier()))
        #
        #         if 'LinearDiscriminantAnalysis' in names:
        #             models.append(('LinearDiscriminant', discriminant_analysis.LinearDiscriminantAnalysis()))
        #         if 'QuadraticDiscriminantAnalysis' in names:
        #             models.append(('QuadraticDiscriminant', discriminant_analysis.QuadraticDiscriminantAnalysis()))
        #
        #         if 'XGBClassifier' in names:
        #             models.append(('XGB', XGBClassifier()))

    print('Количество возвращенных моделей: ', len(return_models), '\n')
    return return_models


# --------------------------------------------------------------------------------------------
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# --------------------------------------------------------------------------------------------
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5, shuffle=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.shuffle =shuffle

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=156)

        oof_pred = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for tr_idx, te_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)

                instance.fit(X[tr_idx], y[tr_idx])
                y_pred = instance.predict(X[te_idx])

                oof_pred[te_idx, i] = y_pred

        self.meta_model_.fit(oof_pred, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# --------------------------------------------------------------------------------------------
# Does VotingRegressot exist ???
def hard_voting(models, X, y, cv_split):
    grid_hard = ensemble.VotingClassifier(estimators=models, voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, X, y, cv=cv_split)
    grid_hard.fit(X, y)

    print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".
          format(grid_hard_cv['train_score'].mean() * 100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".
          format(grid_hard_cv['test_score'].mean() * 100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".
          format(grid_hard_cv['test_score'].std() * 100 * 3))
    print('-' * 10)


# --------------------------------------------------------------------------------------------
def soft_voting(models, X, y, cv_split):
    grid_soft = ensemble.VotingClassifier(estimators=models, voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, X, y, cv=cv_split)
    grid_soft.fit(X, y)

    print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".
          format(grid_soft_cv['train_score'].mean() * 100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".
          format(grid_soft_cv['test_score'].mean() * 100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".
          format(grid_soft_cv['test_score'].std() * 100 * 3))
    print('-' * 10)
