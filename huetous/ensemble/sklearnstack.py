# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import warnings
import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import has_fit_parameter
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.externals import six
import time
from sklearn import model_selection


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class StackingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 estimators=None,
                 regression=True,
                 transform_target=None,
                 transform_pred=None,
                 variant='A',
                 needs_proba=False,
                 metric=None,
                 n_folds=4,
                 stratified=False,
                 shuffle=False,
                 random_state=0,
                 verbose=0):
        self.estimators = estimators
        self.regression = regression
        self.transform_target = transform_target
        self.transform_pred = transform_pred
        self.variant = variant
        self.needs_proba = needs_proba
        self.metric = metric
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    # -------------------------------------------
    # -------------------------------------------
    def fit(self, X, y, sample_weight=None):
        # -------------------------------------------
        # Check parameters
        # -------------------------------------------
        X, y = check_X_y(X, y,
                         accept_sparse=['csr'],
                         force_all_finite=True,
                         multi_output=False)

        if sample_weight is not None:
            X, sample_weight = check_X_y(X, sample_weight,
                                         accept_sparse=['csr'],
                                         force_all_finite=True,
                                         multi_output=False)

        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError('List of estimators is empty.')
        else:
            # Clone
            self.estimators_ = [(name, clone(estim)) for name, estim in self.estimators]
            names, estims = zip(*self.estimators_)
            self._validate_names(names)
            if sample_weight is not None:
                for name, estim in self.estimators_:
                    if not has_fit_parameter(estim, 'sample_weight'):
                        raise ValueError('Underlying estimator [%s] does not support sample weights.' % name)

        if self.variant not in ['A', 'B']:
            raise ValueError('Parameter <variant> is incorrect.')

        if not isinstance(self.n_folds, int):
            raise ValueError('Parameter <n_folds> must be integer.')

        if not self.n_folds > 1:
            raise ValueError('Parameter <n_folds> must be not less than 2.')

        if self.verbose not in [0, 1, 2]:
            raise ValueError('Parameter <verbose> must be 0, 1, or 2.')

        if self.regression and (self.needs_proba or self.stratified):
            warn_str = ('This is regression task hence classification-specific'
                        'parameters set to True will be ignored:')
            if self.needs_proba:
                self.needs_proba = False
                warn_str += '<needs_proba>'
            if self.random_state:
                self.stratified = False
                warn_str += '<strarified>'
            warnings.warn(warn_str, UserWarning)
        # -------------------------------------------
        # Compute attributes
        # -------------------------------------------
        self.train_shape_ = X.shape
        self.n_train_examples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        if not self.regression:
            self.n_classes_ = len(np.unique(y))
        else:
            self.n_classes_ = None
        self.n_estimators_ = len(self.estimators_)
        self.train_footprint_ = self._get_footprint(X)
        # -------------------------------------------
        # Specify metric
        # -------------------------------------------
        if self.metric is None and self.regression:
            self.metric_ = mean_absolute_error
        elif self.metric is None and not self.regression:
            if self.needs_proba:
                self.metric_ = log_loss
            else:
                self.metric_ = accuracy_score
        else:
            self.metric_ = self.metric
        # -------------------------------------------
        # Create report header
        # -------------------------------------------
        if self.verbose > 0:
            if self.regression:
                task_str = 'task:         [regression]'
            else:
                task_str = 'task:         [classification]'
                n_classes_str = 'n_classes:    [%d]' % self.n_classes_
            metric_str = 'metric:       [%s]' % self.metric_.__name__
            variant_str = 'variant:      [%s]' % self.variant
            n_estimators_str = 'n_estimators: [%d]' % self.n_estimators_

            print('=' * 60)
            print(task_str)
            if not self.regression:
                print(n_classes_str)
            print(metric_str)
            print(variant_str)
            print(n_estimators_str + '\n')
        # -------------------------------------------
        # Initialize cross-validation split
        # -------------------------------------------
        if not self.regression and self.stratified:
            self.kf_ = StratifiedKFold(n_splits=self.n_folds,
                                       shuffle=self.shuffle,
                                       random_state=self.random_state)
            self._y_ = y.copy()
        else:
            self.kf_ = KFold(n_splits=self.n_folds,
                             shuffle=self.shuffle,
                             random_state=self.random_state)
            self._y_ = None
        # -------------------------------------------
        # Compute number of classes
        # -------------------------------------------
        if not self.regression and self.needs_proba:
            self.n_classes_implicit_ = len(np.unique(y))
            self.action_ = 'predict_proba'
        else:
            self.n_classes_implicit_ = 1
            self.action_ = 'predict'
        # -------------------------------------------
        # Create empty arrays for OOF
        # -------------------------------------------
        S_train = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))
        # -------------------------------------------
        # Clone estimators for fitting and storing
        # -------------------------------------------
        self.models_A_ = []
        self.models_B_ = None

        for n, est in self.estimators_:
            self.models_A_.append([clone(est) for _ in range(self.n_folds)])

        if self.variant in ['B']:
            self.models_B_ = [clone(est) for n, est in self.estimators_]
        # -------------------------------------------
        # Create emply list to store ame, mean and std for each estimator and each fold
        # -------------------------------------------
        self.scores_ = np.zeros((self.n_estimators_, self.n_folds))
        # -------------------------------------------
        # Create emply list to store ame, mean and std for each estimator
        # -------------------------------------------
        self.mean_std_ = []
        # -------------------------------------------
        # Loop across estimators
        # -------------------------------------------
        for estimator_counter, (name, estimator) in enumerate(self.estimators_):
            if self.verbose > 0:
                estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                print(estimator_str)
            # -------------------------------------------
            # Loop across folds
            # -------------------------------------------
            for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, y)):
                X_tr = X[tr_index]
                y_tr = y[tr_index]
                X_te = X[te_index]
                y_te = y[te_index]

                if sample_weight is not None:
                    sample_weight_tr = sample_weight[tr_index]

                else:
                    sample_weight_tr = None

                _ = self._estimator_action(self.models_A_[estimator_counter][fold_counter],
                                           X_tr, y_tr, None,
                                           sample_weight=sample_weight_tr,
                                           action='fit',
                                           transform=self.transform_target)

                if self.action_ == 'predict_proba':
                    col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                estimator_counter * self.n_classes_implicit_
                                                + self.n_classes_implicit_)
                else:
                    col_slice_estimator = estimator_counter
                S_train[te_index, col_slice_estimator] = self._estimator_action(
                    self.models_A_[estimator_counter][fold_counter],
                    None, None, X_te, action=self.action_, transform=self.transform_pred)

                score = self.metric_(y_te, S_train[te_index, col_slice_estimator])
                self.scores_[estimator_counter, fold_counter] = score

                if self.verbose > 1:
                    fold_str = '    fold %2d:   [%.8f]' % (fold_counter, score)
                    print(fold_str)

            estim_name = self.estimators_[estimator_counter][0]
            estim_mean = np.mean(self.scores_[estimator_counter])
            estim_std = np.std(self.scores_[estimator_counter])
            self.mean_std_.append((estim_name, estim_mean, estim_std))

            if self.verbose > 1:
                sep_str = '    ----'
                print(sep_str)

            if self.verbose > 0:
                mean_str = '    MEAN:       [%.8f] +/- [%.8f]\n' % (estim_mean, estim_std)
                print(mean_str)

            if self.variant in ['B']:
                if self.verbose > 0:
                    print('     Fitting on full train set...\n')
                _ = self._estimator_action(self.models_B_[estimator_counter],
                                           X, y, None,
                                           sample_weight=sample_weight,
                                           action='fit',
                                           transform=self.transform_target)
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        return self

    # -------------------------------------------
    # -------------------------------------------
    def fit_transform(self, X, y, sample_weight=None):
        # Fit all base estimators and transform train set
        return self.fit(X, y, sample_weight).transform(X)

    # -------------------------------------------
    # -------------------------------------------
    def transform(self, X, is_train_set=None):
        check_is_fitted(self, ['models_A_'])
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)

        if is_train_set is None:
            is_train_set = self._check_identity(X)

        if self.verbose > 0:
            print('=' * 60)
            if is_train_set:
                print('Train set was detected.')
            print('Transforming...\n')

        # -------------------------------------------
        # Transform train set
        # -------------------------------------------
        if is_train_set:
            if self.train_shape_ != X.shape:
                raise ValueError('Train set must have the same shape'
                                 'in order to be transformed.')
            S_train = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

            # -------------------------------------------
            # Loop across estimators
            # -------------------------------------------

            for estimator_counter, (name, estimator) in enumerate(self.estimators_):
                if self.verbose > 0:
                    estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                    print(estimator_str)

                # -------------------------------------------
                # Loop across folds
                # -------------------------------------------
                for fold_counter, (tr_index, te_index) in enumerate(self.kf_.split(X, self._y_)):
                    X_te = X[te_index]

                    if self.action_ == 'predict_proba':
                        col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                    estimator_counter * self.n_classes_implicit_
                                                    + self.n_classes_implicit_)
                    else:
                        col_slice_estimator = estimator_counter
                    S_train[te_index, col_slice_estimator] = self._estimator_action(
                        self.models_A_[estimator_counter][fold_counter],
                        None, None, X_te, action=self.action_, transform=self.transform_pred)

                    if self.verbose > 1:
                        fold_str = '    model from fold %2d: done' % fold_counter
                        print(fold_str)

                if self.verbose > 1:
                    sep_str = '     ----'
                    print(sep_str)

                if self.verbose > 0:
                    done_str = '     DONE\n'
                    print(done_str)

            if not self.regression and not self.needs_proba:
                S_train = S_train.astype(int)

            return S_train
        # -------------------------------------------
        # Transform any other set
        # -------------------------------------------
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError('Inconsistent number of features.')
            S_test = np.zeros((X.shape[0], self.n_estimators_ * self.n_classes_implicit_))

            # -------------------------------------------
            # Loop across estimators
            # -------------------------------------------
            for estimator_counter, (name, estimator) in enumerate(self.estimators_):
                if self.verbose > 0:
                    estimator_str = 'estimator %2d: [%s: %s]' % (estimator_counter, name, estimator.__class__.__name__)
                    print(estimator_str)

                # -------------------------------------------
                # Variant A
                # -------------------------------------------
                if self.variant in ['A']:
                    S_test_temp = np.zeros((X.shape[0], self.n_folds * self.n_classes_implicit_))
                    # -------------------------------------------
                    # Loop across fitted models (same as loop across folds)
                    # -------------------------------------------
                    for fold_counter, model in enumerate(self.models_A_[estimator_counter]):
                        if self.action_ == 'predict_proba':
                            col_slice_fold = slice(fold_counter * self.n_classes_implicit_,
                                                   fold_counter * self.n_classes_implicit_
                                                   + self.n_classes_implicit_)
                        else:
                            col_slice_fold = fold_counter
                        S_test_temp[:, col_slice_fold] = self._estimator_action(model, None, None, X,
                                                                                action=self.action_,
                                                                                transform=self.transform_pred)

                        if self.verbose > 1:
                            fold_str = '    model from fold %2d: done' % fold_counter
                            print(fold_str)

                    if self.verbose > 1:
                        sep_str = '    ----'
                        print(sep_str)

                    if self.action_ == 'predict_proba':
                        for class_id in range(self.n_classes_implicit_):
                            S_test[:, estimator_counter * self.n_classes_implicit_ + class_id] = \
                                np.mean(S_test_temp[:, class_id::self.n_classes_implicit_], axis=1)
                    else:
                        if self.regression:
                            S_test[:, estimator_counter] = np.mean(S_test_temp, axis=1)
                        else:
                            S_test[:, estimator_counter] = st.mode(S_test_temp, axis=1)[0].ravel()
                    if self.verbose > 0:
                        done_str = '    DONE\n'
                        print(done_str)

                # -------------------------------------------
                # Variant B
                # -------------------------------------------
                else:
                    if self.action_ == 'predict_proba':
                        col_slice_estimator = slice(estimator_counter * self.n_classes_implicit_,
                                                    estimator_counter * self.n_classes_implicit_
                                                    + self.n_classes_implicit_)
                    else:
                        col_slice_estimator = estimator_counter
                    S_test[:, col_slice_estimator] = self._estimator_action(self.models_B_[estimator_counter],
                                                                            None, None, X,
                                                                            action=self.action_,
                                                                            transform=self.transform_pred)

                    if self.verbose > 0:
                        done_str = '     DONE\n'
                        print(done_str)

            if not self.regression and not self.needs_proba:
                S_test = S_test.astype(int)

            return S_test

    # -------------------------------------------
    # -------------------------------------------
    # -------------------------------------------
    # -------------------------------------------
    def _transformer(self, y, func=None):
        if func is None:
            return y
        else:
            return func(y)

    # -------------------------------------------
    # -------------------------------------------
    def _estimator_action(self, estimator, X_train, y_train, X_test,
                          sample_weight=None, action=None, transform=None):
        if action == 'fit':
            if sample_weight is not None:
                return estimator.fit(X_train, self._transformer(y_train, func=transform), sample_weigth=sample_weight)
            else:
                return estimator.fit(X_train, self._transformer(y_train, func=transform))
        elif action == 'predict':
            return self._transformer(estimator.predict(X_test), func=transform)
        elif action == 'predict_proba':
            return self._transformer(estimator.predict_proba(X_test), func=transform)
        else:
            raise ValueError('Parameter action is incorrect.')

    # -------------------------------------------
    # -------------------------------------------
    def _get_footprint(self, X, n_items=1000):
        try:
            footprint = []
            r, c = X.shape
            n = r * c
            ids = np.random.choice(n, min(n_items, n), replace=True)

            for i in ids:
                row = i // c
                col = i - row * c
                footprint.append((row, col, X[row, col]))
            return footprint
        except Exception:
            raise ValueError('Internal error.')

    # -------------------------------------------
    # -------------------------------------------
    def _check_identity(self, X, rtol=1e-05, atol=1e-08, equal_nan=False):
        try:
            if X.shape != self.train_shape_:
                return False
            try:
                for coo in self.train_footprint_:
                    assert np.isclose(X[coo[0], coo[1]], coo[2], rtol=rtol, atol=atol, equal_nan=equal_nan)
                    return True
            except AssertionError:
                return False

        except Exception:
            raise ValueError('Iternal error.')

    # -------------------------------------------
    # -------------------------------------------
    def _get_params(self, attr, deep=True):
        out = super(StackingTransformer, self).get_params(deep=False)
        if not deep:
            return out
        estimators = getattr(self, attr)
        if estimators is None:
            return out
        out.update(estimators)
        for name, estimator in estimators:
            for key, value in six.iteritems(estimators.get_params(deep=True)):
                out['%s__%s' % (name, key)] = value
        return out

    # -------------------------------------------
    # -------------------------------------------
    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Estimator names are not unique.')
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor argumesnts:'
                             '%s' % sorted(invalid_names))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got %s' % invalid_names)

    # -------------------------------------------
    # -------------------------------------------
    def is_train_set(self, X):
        check_is_fitted(self, ['models_A_'])
        X = check_array(X, accept_sparse=['csr'], force_all_finite=True)
        return self._check_identity(X)


