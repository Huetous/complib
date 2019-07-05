import sklearn
import xgboost
import lightgbm
import catboost
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np

# ------------------------------------------------------------------------
xgb_space = {
    'max_depth': hp.quniform("max_depth", 4, 7, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'subsample': hp.uniform('subsample', 0.75, 1.0),
    # 'gamma': hp.uniform('gamma', 0.0, 0.5),
    'gamma': hp.loguniform('gamma', -5.0, 0.0),
    # 'eta': hp.uniform('eta', 0.005, 0.018),
    'eta': hp.loguniform('eta', -4.6, -2.3),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.70, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.70, 1.0),
}

lgb_space = {
    'num_leaves': hp.quniform('num_leaves', 10, 200, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 200, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
    'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
    'max_bin': hp.quniform('max_bin', 64, 512, 1),
    'bagging_freq': hp.quniform('bagging_freq', 1, 5, 1),
    'lambda_l1': hp.uniform('lambda_l1', 0, 10),
    'lambda_l2': hp.uniform('lambda_l2', 0, 10),
}

cat_space = {
    'depth': hp.quniform("depth", 4, 7, 1),
    'rsm': hp.uniform('rsm', 0.75, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -3.0, -0.7),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
}


# ------------------------------------------------------------------------
class xgb_hyperopt():
    def __init__(self, X_tr, y_tr, X_val, y_val, X_te, y_te,
                 n_probes=500, algo=tpe.suggest,
                 eval_metric='log_loss', objective='binary:logistic',
                 num_class=10, num_boost_round=5000,
                 early_stopping_rounds=200):
        self.n_probes = n_probes
        self.algo = algo

        self.dtrain = xgboost.DMatrix(X_tr, y_tr)
        self.dval = xgboost.DMatrix(X_val, y_val)
        self.dtest = xgboost.DMatrix(X_te, y_te)
        self.X_te = X_te
        self.y_te = y_te

        self.watchlist = [(self.dtrain, 'train'), (self.dtest, 'test')]
        self.eval_metric = eval_metric
        self.objective = objective
        self.num_class = num_class
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.obj_call_count = 0
        self.cur_best_loss = np.inf

    def get_params(self, space):
        params = dict()
        params['booster'] = space['booster'] if 'booster' in space else 'gbtree'
        params['subsample'] = space['subsample']
        params['max_depth'] = int(space['max_depth'])
        params['seed'] = space['seed'] if 'seed' in space else 1337
        params['min_child_weight'] = space['min_child_weight']
        params['eta'] = space['eta']
        params['gamma'] = space['gamma'] if 'gamma' in space else 0.01
        params['colsample_bytree'] = space['colsample_bytree']
        params['colsample_bylevel'] = space['colsample_bylevel']

        params['scale_pos_weight'] = 1.0
        params['eval_metric'] = self.eval_metric
        params['objective'] = self.objective
        params['silent'] = 1
        params['num_class'] = self.num_class
        params['update'] = 'grow_gpu'
        return params

    def obj(self, space):
        self.obj_call_count += 1
        print('XGB. Objective call #{}, current best loss: {.8f}'.
              format(self.obj_call_count, self.cur_best_loss))

        params = self.get_params(space)
        model = xgboost.train(params=params,
                              dtrain=self.dtrain,
                              num_boost_round=self.num_boost_round,
                              evals=self.watchlist,
                              verbose_eval=False,
                              early_stopping_rounds=self.early_stopping_rounds)

        print('nb_trees:{}, val_loss: {:7.5f}'.
              format(model.best_ntree_limit, model.best_score))

        y_pred = model.predict(self.dtest, ntree_limit=model.best_ntree_limit)
        test_loss = sklearn.metrics.log_loss(self.y_te, y_pred, labels=list(range(10)))

        acc = sklearn.metrics.accuracy_score(self.y_te, np.argmax(y_pred, axis=1))
        print('test_loss={} test_acc={}'.format(test_loss, acc))

        if test_loss < self.cur_best_loss:
            self.cur_best_loss = test_loss
            print('New best loss: {}'.format(self.cur_best_loss))

        return {'loss': test_loss, 'status': STATUS_OK}

    def run(self):
        trials = Trials()
        best = hyperopt.fmin(fn=self.obj,
                             space=xgb_space,
                             algo=self.algo,
                             max_evals=self.n_probes,
                             trials=trials,
                             verbose=1)
        print('Best params: ', best)
        return best


# ------------------------------------------------------------------------
class lgb_hyperopt():
    def __init__(self, X_tr, y_tr, X_val, y_val, X_te, y_te,
                 max_depth=-1, application='multiclass',
                 num_class=10, metric='multi_logloss',
                 num_boost_round=10000,
                 early_stopping_rounds=200,
                 n_probes=500, algo=tpe.suggest):
        self.dtrain = lightgbm.Dataset(X_tr, y_tr)
        self.dval = lightgbm.Dataset(X_val, y_val)
        self.X_te = X_te
        self.y_te = y_te

        self.n_probes = n_probes
        self.algo = algo

        self.max_depth = max_depth
        self.application = application
        self.num_class = num_class
        self.metric = metric
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.obj_call_count = 0
        self.cur_best_loss = np.inf

    def get_params(self, space):
        params = dict()
        params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
        params['learning_rate'] = space['learning_rate']
        params['num_leaves'] = int(space['num_leaves'])
        params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
        params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
        params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
        params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
        params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
        params['feature_fraction'] = space['feature_fraction']
        params['bagging_fraction'] = space['bagging_fraction']
        params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

        params['max_depth'] = self.max_depth
        params['num_class'] = self.num_class
        params['application'] = self.application
        params['metric'] = self.metric
        return params

    def obj(self):
        self.obj_call_count += 1
        print('\nLGB. Objective call #{}, current best loss: {:8f}'.
              format(self.obj_call_count, self.cur_best_loss))

        params = self.get_params(space=lgb_space)
        model = lightgbm.train(params=params,
                               train_set=self.dtrain,
                               num_boost_round=self.num_boost_round,
                               valid_sets=self.dval,
                               early_stopping_rounds=self.early_stopping_rounds,
                               verbose_eval=False)

        n_trees = model.best_iteration
        val_loss = model.best_score
        print('n_trees: {}, val_loss: {}'.format(n_trees, val_loss))

        y_pred = model.predict(self.X_te, num_iteration=n_trees)
        test_loss = sklearn.metrics.log_loss(self.y_te, y_pred, labels=list(range(10)))
        acc = sklearn.metrics.accuracy_score(self.y_te, np.argmax(y_pred, axis=1))
        print('test_loss={} test_acc={}'.format(test_loss, acc))

        if test_loss < self.cur_best_loss:
            self.cur_best_loss = test_loss
            print('New best loss: {}'.format(self.cur_best_loss))

        return {'loss': test_loss, 'status': STATUS_OK}

    def run(self):
        trials = Trials()
        best = hyperopt.fmin(fn=self.obj,
                             space=xgb_space,
                             algo=self.algo,
                             max_evals=self.n_probes,
                             trials=trials,
                             verbose=1)
        print('Best params: ', best)
        return best


# ------------------------------------------------------------------------
class cat_hyperopt():
    def __init__(self, X_tr, y_tr, X_val, y_val, X_te, y_te,
                 max_depth=-1, application='multiclass',
                 num_class=10, metric='multi_logloss',
                 num_boost_round=10000,
                 early_stopping_rounds=200,
                 n_probes=500, algo=tpe.suggest):
        self.dtrain = catboost.Pool(X_tr, y_tr)
        self.dval = catboost.Pool(X_val, y_val)

        self.X_te = X_te
        self.y_te = y_te

        self.n_probes = n_probes
        self.algo = algo

        self.max_depth = max_depth
        self.application = application
        self.num_class = num_class
        self.metric = metric
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.obj_call_count = 0
        self.cur_best_loss = np.inf

    def get_params(self, space):
        params = dict()
        params['learning_rate'] = space['learning_rate']
        params['depth'] = int(space['depth'])
        params['l2_leaf_reg'] = space['l2_leaf_reg']
        params['rsm'] = space['rsm']

    def obj(self):
        self.obj_call_count += 1

        print('\nCatBoost. Objective call #{}, current best loss: {:8f}'.
              format(self.obj_call_count, self.cur_best_loss))

        params = self.get_params(cat_space)
        model = catboost.CatBoostClassifier(iterations=self.num_boost_round,
                                            learning_rate=params['learning_rate'],
                                            depth=int(params['depth']),
                                            loss_function=self.metric,
                                            use_best_model=True,
                                            eval_metric=self.metric,
                                            l2_leaf_reg=params['l2_leaf_reg'],
                                            random_seed=params['seed'],
                                            verbose=False)
        model.fit(self.dtrain, eval_set=self.dval, verbose=False)

        y_pred = model.predict_proba(self.X_te)

        test_loss = sklearn.metrics.log_loss(self.y_te, y_pred, labels=list(range(10)))
        acc = sklearn.metrics.accuracy_score(self.y_te, np.argmax(y_pred, axis=1))

        if test_loss < self.cur_best_loss:
            self.cur_best_loss = test_loss
            print('New best loss: {}'.format(self.cur_best_loss))

        return {'loss': test_loss, 'status': STATUS_OK}

    def run(self):
        trials = Trials()
        best = hyperopt.fmin(fn=self.obj,
                             space=xgb_space,
                             algo=self.algo,
                             max_evals=self.n_probes,
                             trials=trials,
                             verbose=1)
        print('Best params: ', best)
        return best
# ------------------------------------------------------------------------
