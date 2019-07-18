# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import os
import sys
import warnings
from datetime import datetime
import numpy as np
import scipy.stats as st
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.base import clone


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def transformer(y, func=None):
    if func is None:
        return y
    else:
        return func(y)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def model_action(model, X_train, y_train, X_test,
                 sample_weight=None, action=None, transform=None):
    if action == 'fit':
        if sample_weight is not None:
            return model.fit(X_train, transformer(y_train, func=transform), sample_weight=sample_weight)
        else:
            return model.fit(X_train, transformer(y_train, func=transform))
    elif action == 'predict':
        return transformer(model.predict(X_test), func=transform)
    elif action == 'predict_proba':
        return transformer(model.predict_proba(X_test), func=transform)
    else:
        raise ValueError('Parameter action is not set or incorrect.')


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def model_params(model):
    string = ''

    if hasattr(model, 'get_params'):
        params_dict = model.get_params()
        maxlen = 0
        for key in params_dict:
            if len(key) > maxlen:
                maxlen = len(key)

        sorted_keys = sorted(params_dict.keys())
        for key in sorted_keys:
            string += '%-*s %s\n' % (maxlen, key, params_dict[key])

    elif hasattr(model, '__repr__'):
        string = model.__repr__()
        string += '\n'

    else:
        string = 'Model has no ability to show parameters.\n'

    return string


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

def stack(models, X_train, y_train, X_test,
          sample_weight=None, regression=True,
          transform_target=None, transform_pred=None,
          mode='oof_pred_bag', needs_proba=False, save_dir=None,
          metric=None, n_folds=4, stratified=False,
          shuffle=False, random_state=0, verbose=0):
    # -------------------------------------
    # Check if values are correct
    # -------------------------------------
    if len(models) == 0:
        raise ValueError('List of models if empty.')

    if mode not in ['pred', 'pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
        raise ValueError('Parameter <mode> is incorrect.')

    if save_dir is not None:
        save_dir = os.path.normpath(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError('Path does not exist or is nor a directory. Check <save_dir> parameter.')

    if not isinstance(n_folds, int):
        raise ValueError('Parameter <n_folds> must be integer.')

    if not n_folds > 1:
        raise ValueError('Parameter <n_folds> must be not less than 2.')

    if verbose not in [0, 1, 2]:
        raise ValueError('Parameter <verbose> must be 0, 1, or 2.')

    X_train, y_train = check_X_y(X_train, y_train,
                                 accept_sparse=['csr'],
                                 force_all_finite=False,
                                 allow_nd=True,
                                 multi_output=False)
    if X_test is not None:
        X_test = check_array(X_test,
                             accept_sparse=['csr'],
                             allow_nd=True,
                             force_all_finite=False)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight).ravel()

    regression = bool(regression)
    stratified = bool(stratified)
    needs_proba = bool(needs_proba)
    shuffle = bool(shuffle)
    # -------------------------------------
    # Check for inapplicable parameter combinations
    # -------------------------------------
    if regression and (needs_proba or stratified):
        warn_str = 'This is regression task hense classification-specific parameters set to <True> will be ignored:'
        if needs_proba:
            needs_proba = False
            warn_str += '<needs_proba>'
        if stratified:
            stratified = False
            warn_str += '<stratified>'
        warnings.warn(warn_str, UserWarning)
    # -------------------------------------
    # Specify metric
    # -------------------------------------
    if metric is None and regression:
        metric = mean_absolute_error
    elif metric is None and not regression:
        if needs_proba:
            metric = log_loss
        else:
            metric = accuracy_score
    # -------------------------------------
    # Create report header
    # -------------------------------------
    if save_dir is not None or verbose > 0:
        if regression:
            task_str = 'task:           [regression]'
        else:
            task_str = 'task:           [classification]'
            n_classes_str = 'n_classes:      [%d]' % len(np.unique(y_train))
        metric_str = 'metric:           [%s]' % metric.__name__
        mode_str = 'mode:           [%s]' % mode
        n_models_str = 'n_models:           [%d]' % len(models)

    if verbose > 0:
        print(task_str)
        if not regression:
            print(n_classes_str)
        print(metric_str)
        print(mode_str)
        print(n_models_str + '\n')
    # -------------------------------------
    # Split indices to get folds
    # -------------------------------------
    if not regression and stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    # -------------------------------------
    # Compute number of classes
    # -------------------------------------
    if not regression and needs_proba:
        n_classes = len(np.unique(y_train))
        action = 'predict_proba'
    else:
        n_classes = 1
        action = 'predict'
    # -------------------------------------
    # Create empty numpy arrays for OOF
    # -------------------------------------
    if mode in ['oof_pred', 'oof_pred_bag']:
        S_train = np.zeros((X_train.shape[0], len(models) * n_classes))
        S_test = np.zeros((X_test.shape[0], len(models) * n_classes))
    elif mode in ['oof']:
        S_train = np.zeros((X_train.shape[0], len(models) * n_classes))
        S_test = None
    elif mode in ['pred', 'pred_bag']:
        S_train = None
        S_test = np.zeros((X_test.shape[0], len(models) * n_classes))
    # -------------------------------------
    # -------------------------------------
    models_folds_str = ''
    # -------------------------------------
    # Loop across models
    # -------------------------------------
    for model_counter, model in enumerate(models):
        if save_dir is not None or verbose > 0:
            model_str = 'model %2d:     [%s]' % (model_counter, model.__class__.__name__)
        if save_dir is not None:
            models_folds_str += '-' * 40 + '\n'
            models_folds_str += model_str + '\n'
            models_folds_str += '-' * 40 + '\n\n'
            models_folds_str += model_params(model)
        if verbose > 0:
            print(model_str)

        if mode in ['pred_bag', 'oof_pred_bag']:
            S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))

        scores = np.array([])

        # -------------------------------------
        # Loop across folds
        # -------------------------------------
        if mode in ['pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
            for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
                X_tr = X_train[tr_index]
                y_tr = y_train[tr_index]
                X_te = X_train[te_index]
                y_te = y_train[te_index]

                # Split sample weights accordingly
                if sample_weight is not None:
                    sample_weight_tr = sample_weight[tr_index]
                else:
                    sample_weight_tr = None

                # Clone to avoid fitting model directly inside users list
                model = clone(model, safe=False)

                # Fit 1-st level model
                if mode in ['pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
                    _ = model_action(model, X_tr, y_tr, None,
                                     sample_weight=sample_weight_tr, action='fit',
                                     transform=transform_target)

                # Predict ouf-of-fold part of train set
                if mode in ['oof', 'oof_pred', 'oof_pred_bag']:
                    if action == 'predict_proba':
                        col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                    else:
                        col_slice_model = model_counter
                    S_train[te_index, col_slice_model] = model_action(model,
                                                                      None, None, X_te, action=action,
                                                                      transform=transform_pred)
                # Predict full test set in each fold
                if mode in ['pred_bag', 'oof_pred_bag']:
                    if action == 'predict_proba':
                        col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
                    else:
                        col_slice_fold = fold_counter
                    S_test_temp[:, col_slice_fold] = model_action(model, None, None, X_test, action=action,
                                                                  transform=transform_pred)

                # Compute scores
                if mode in ['oof', 'oof_pred', 'oof_pred_bag']:
                    if save_dir is not None or verbose > 0:
                        score = metric(y_te, S_train[te_index, col_slice_model])
                        scores = np.append(scores, score)
                        fold_str = '        fold %2d:   [%.8f]' % (fold_counter, score)
                    if save_dir is not None:
                        models_folds_str += fold_str + '\n'
                    if verbose > 1:
                        print(fold_str)

        # Compute mean or mode of predictions for test set in bag modes
        if mode in ['pred_bag', 'oof_pred_bag']:
            if action == 'predict_proba':
                # Compute means of probabilities for each class
                for class_id in range(n_classes):
                    S_test[:, model_counter * n_classes + class_id] = np.mean(S_test_temp[:, class_id::n_classes],
                                                                              axis=1)
            else:
                if regression:
                    S_test[:, model_counter] = np.mean(S_test_temp, axis=1)
                else:
                    S_test[:, model_counter] = st.mode(S_test_temp, axis=1)[0].ravel()

        # Compute scores: mean + std and full
        if mode in ['oof', 'oof_pred', 'oof_pred_bag']:
            if save_dir is not None or verbose > 0:
                sep_str = '        ----'
                mean_str = '        MEAN:       [%.8f] +/- [%.8f]' % (np.mean(scores), np.std(scores))
                full_str = '        FULL:       [%.8f]\n' % (metric(y_train, S_train[:, col_slice_model]))
            if save_dir is not None:
                models_folds_str += sep_str + '\n'
                models_folds_str += mean_str + '\n'
                models_folds_str += full_str + '\n'
            if verbose > 0:
                print(sep_str)
                print(mean_str)
                print(full_str)

        # Fit model on full train set and predict test set
        if mode in ['pred', 'oof_pred']:
            if verbose > 0:
                print('     Fitting on full train set...\n')
            _ = model_action(model, X_train, y_train, None, sample_weight=sample_weight,
                             action='fit', transform=transform_target)
            if action == 'predict_proba':
                col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
            else:
                col_slice_model = model_counter
            S_test[:, col_slice_model] = model_action(model, None, None, X_test, action=action
                                                      , transform=transform_pred)
    # -------------------------------------
    # Cast class labels to int
    # -------------------------------------
    if not regression and not needs_proba:
        if S_train is not None:
            S_train = S_train.astype(int)
        if S_test is not None:
            S_test = S_test.astype(int)
    # -------------------------------------
    # Save OOF and log
    # -------------------------------------
    if save_dir is not None:
        try:
            time_str = datetime.now().strftime('[%Y.%m.%d].[%H.%M.%S].%f') + ('.%06x' % np.random.randint(0xffffff))

            file_name = time_str + '.npy'
            log_file_name = time_str + '.log.txt'

            full_path = os.path.join(save_dir, file_name)
            log_full_path = os.path.join(save_dir, log_file_name)

            np.save(full_path, np.array([S_train, S_test]))

            log_str = 'huestack log '
            log_str += time_str + '\n\n'
            log_str += task_str + '\n'
            if not regression:
                log_str += n_classes_str + '\n'
            log_str += metric_str + '\n'
            log_str += mode_str + '\n'
            log_str += n_models_str + '\n'
            log_str += models_folds_str + '\n'
            log_str += '-' * 40 + '\n'
            log_str += 'END\n'
            log_str += '-' * 40 + '\n'

            with open(log_full_path, 'w') as f:
                _ = f.write(log_str)

            if verbose > 0:
                print('Result was saved to [%s]' % full_path)
        except:
            print('Error while saving files: \n%s' % sys.exc_info()[1])

    return (S_train, S_test)
