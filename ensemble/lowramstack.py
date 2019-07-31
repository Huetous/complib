# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import os
import sys
import pandas as pd
import warnings
from datetime import datetime
import numpy as np
import scipy.stats as st
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.base import clone


def aligned_text(str1, str2, MARGIN=20):
    str = str1
    str += ' ' * (MARGIN - len(str))
    str += str2
    return str


# ------------------------------------------------------------------------

def model_action(model, X_train, y_train, X_test, X_valid=None, y_valid=None,
                 sample_weight=None, action=None, is_custom=False):
    if action is 'fit':
        if is_custom:
            if X_valid is None or y_valid is None:
                raise ValueError('Parameters <X_valid> and <y_valid> must be specified for custom models.')
            return model.fit(X_train, y_train, X_valid, y_valid)
        else:
            return model.fit(X_train, y_train)
    elif action is 'predict':
        return model.predict(X_test)
    elif action is 'predict_proba':
        return model.predict_proba(X_test)
    else:
        raise ValueError('Parameter <action> is not set or incorrect.')


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
def stack(models, X_train, y_train, X_test,
          regression=True, metrics=None,
          mode='oof_pred_bag', needs_proba=False, save_dir=None,
          cv_schema='kf', n_folds=4, shuffle=False, random_state=0,
          verbose=0):
    # -------------------------------------
    # Check parameters
    # -------------------------------------
    if len(models) is 0:
        raise ValueError('List of models is empty.')

    if mode not in ['pred', 'pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
        raise ValueError('Parameter <mode> is incorrect.')

    if save_dir is not None:
        save_dir = os.path.normpath(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError('Path does not exist or is nor a directory. Check <save_dir> parameter.')

    if cv_schema not in ['kf', 'skf', 'ts']:
        raise ValueError("Parameter <cv_schema> must be 'kf', 'skf' or 'ts' ")

    if not isinstance(n_folds, int):
        raise ValueError('Parameter <n_folds> must be integer.')

    if not n_folds > 1:
        raise ValueError('Parameter <n_folds> must be not less than 2.')

    if verbose not in [0, 1, 2]:
        raise ValueError('Parameter <verbose> must be 0, 1, or 2.')

    # -------------------------------------
    # Check given data
    # -------------------------------------
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

    regression = bool(regression)
    needs_proba = bool(needs_proba)
    shuffle = bool(shuffle)

    # -------------------------------------
    # Check for inapplicable parameter combinations
    # -------------------------------------

    if regression and (needs_proba or cv_schema is 'skf'):
        warn_str = 'This is regression task hense classification-specific parameters will be ignored:\n'
        if needs_proba:
            needs_proba = False
            warn_str += '<needs_proba> changed to False\n'
        if cv_schema is 'skf':
            warn_str += "<cv_schema> changed to 'kf'\n"
        warnings.warn(warn_str, UserWarning)

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
    # Specify metric
    # -------------------------------------
    if metrics is None:
        metrics = []
        if regression:
            metrics.append(('mae', mean_absolute_error))
        else:
            if needs_proba:
                metrics.append(('log_loss', log_loss))
            else:
                metrics.append(('roc_auc_score', roc_auc_score))

    # -------------------------------------
    # Specify split scheme
    # -------------------------------------

    if cv_schema is 'kf':
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    if not regression and cv_schema is 'skf':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    if cv_schema is 'ts':
        kf = TimeSeriesSplit(n_splits=n_folds)

    # -------------------------------------
    # Create report header
    # -------------------------------------
    if save_dir is not None or verbose > 0:
        if regression:
            task_str = aligned_text('task:', '[regression]')
        else:
            task_str = aligned_text('task:', '[classification]')
            action_str = aligned_text('action:', f'[{action}]')
            n_classes_str = aligned_text('n_classes:', f'[{n_classes}]')
        metrics_str = ''
        for metric_name, metric in metrics:
            metrics_str += f'[{metric_name}]'
        metrics_str = aligned_text('metrics:', metrics_str)
        mode_str = aligned_text('mode:', f'[{mode}]')
        n_models_str = aligned_text('n_models:', f'[{len(models)}]')
        cv_schema_str = aligned_text('cv_schema:', f'[{kf.__class__.__name__}]')

    if verbose > 0:
        print(task_str)
        if not regression:
            print(n_classes_str)
            print(action_str)
        print(metrics_str)
        print(mode_str)
        print(n_models_str)
        print(cv_schema_str + '\n')

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
    # Log string
    # -------------------------------------
    models_folds_str = ''

    # ------------------------------------------------
    # Loop across models
    # ------------------------------------------------
    for model_counter, (model_name, model) in enumerate(models):
        if save_dir is not None or verbose > 0:
            model_str = aligned_text(f'model {model_counter}:', f'[{model_name}]')
        if save_dir is not None:
            models_folds_str += '-' * 40 + '\n'
            models_folds_str += model_str + '\n'
            models_folds_str += '-' * 40 + '\n\n'
            models_folds_str += model_params(model)
        if verbose > 0:
            print(model_str)

        if mode in ['pred_bag', 'oof_pred_bag']:
            S_test_temp = np.zeros((X_test.shape[0], n_folds * n_classes))

        scores = pd.DataFrame()
        for name, metric in metrics:
            scores[name] = np.ndarray((n_folds,))
        # -------------------------------------
        # Custom models requires special treatment while training
        # -------------------------------------
        if model.__class__.__name__ in ['HuetousLGB', 'HuetousXGB', 'HuetousCatBoost']:
            is_custom = True
        else:
            is_custom = False

        # -------------------------------------
        # Loop across folds
        # -------------------------------------
        if mode in ['pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
            for fold_counter, (tr_index, val_index) in enumerate(kf.split(X_train, y_train)):
                X_tr, X_val = X_train[tr_index], X_train[val_index]
                y_tr, y_val = y_train[tr_index], y_train[val_index]

                # Clone to avoid fitting model directly inside users list
                model = clone(model, safe=False)

                # Fit 1-st level model
                if mode in ['pred_bag', 'oof', 'oof_pred', 'oof_pred_bag']:
                    _ = model_action(model=model,
                                     X_train=X_tr, y_train=y_tr, X_valid=X_val, y_valid=y_val,
                                     X_test=None,
                                     action='fit', is_custom=is_custom)

                # Predict ouf-of-fold part of train set
                if mode in ['oof', 'oof_pred', 'oof_pred_bag']:
                    if action is 'predict_proba':
                        col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                    else:
                        col_slice_model = model_counter
                    S_train[val_index, col_slice_model] = model_action(model=model,
                                                                       X_train=None, y_train=None,
                                                                       X_test=X_val,
                                                                       action=action, is_custom=is_custom)
                # Predict full test set in each fold
                if mode in ['pred_bag', 'oof_pred_bag']:
                    if action == 'predict_proba':
                        col_slice_fold = slice(fold_counter * n_classes, fold_counter * n_classes + n_classes)
                    else:
                        col_slice_fold = fold_counter
                    S_test_temp[:, col_slice_fold] = model_action(model=model,
                                                                  X_train=None, y_train=None,
                                                                  X_test=X_test,
                                                                  action=action, is_custom=is_custom)

                # Compute scores
                if mode in ['oof', 'oof_pred', 'oof_pred_bag']:
                    if save_dir is not None or verbose > 0:
                        fold_str = ''
                        for metric_name, metric in metrics:
                            score = metric(y_val, S_train[val_index, col_slice_model])
                            scores.loc[fold_counter, metric_name] = score
                            score_str = aligned_text(f'[{metric_name}]', '[{:.8f}]'.format(score), MARGIN=12)
                            fold_str += '        fold {}:     {}\n'.format(fold_counter, score_str)
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
                sep_str = aligned_text('', '-' * 24, MARGIN=8)
                mean_str = ''
                for col in scores.columns:
                    scores_str = aligned_text(f'[{col}]', '[{:.8f}]'.format(np.mean(scores[col])), MARGIN=12)
                    scores_str += ' +/- [{:.8f}]\n'.format(np.std(scores[col]))
                    mean_str += '        MEAN        {}'.format(scores_str)
            if save_dir is not None:
                models_folds_str += sep_str + '\n'
                models_folds_str += mean_str + '\n'
            if verbose > 0:
                print(sep_str)
                print(mean_str)

        # Fit model on full train set and predict test set
        if mode in ['pred', 'oof_pred']:
            if verbose > 0:
                print('     Fitting on full train set...\n')
            _ = model_action(model=model,
                             X_train=X_train, y_train=y_train,
                             X_test=None,
                             action='fit', is_custom=is_custom)
            if action == 'predict_proba':
                col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
            else:
                col_slice_model = model_counter
            S_test[:, col_slice_model] = model_action(model=model,
                                                      X_train=None, y_train=None,
                                                      X_test=X_test,
                                                      action=action, is_custom=is_custom)
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

            log_str = 'huetous log '
            log_str += time_str + '\n\n'
            log_str += task_str + '\n'
            if not regression:
                log_str += n_classes_str + '\n'
            log_str += metrics_str + '\n'
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

    return S_train, S_test
