import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from category_encoders import BinaryEncoder
import gc


# --------------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------------

# Adds to a dataset new columns which represents parts of datetime
def do_date_extract(df, col):
    new_cols = []
    parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter', 'dayofyear']

    new_df = pd.DataFrame()
    new_df[col] = pd.to_datetime(df[col])
    for p in parts:
        new_df[p] = getattr(df[col].dt, p).astype(int)
        new_cols.append(p)

    print(f'do_date_extract: Done. Added {len(new_cols)} new columns.')
    return new_df, new_cols


# Adds to a dataset new column which notes 1 if column contains nan values
def do_isnull(X, X_test, cols):
    X_tr = pd.DataFrame()
    X_te = pd.DataFrame()
    new_cols = []
    for c in cols:
        c_name = c + '_nan'
        new_cols.append(c_name)
        X_tr[c_name] = np.where(X[c].isna(), 0, 1)
        X_te[c_name] = np.where(X_test[c].isna(), 0, 1)
    print(f'do_isnull: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# --------------------------------------------------------------------------------------------
# Categorical
# --------------------------------------------------------------------------------------------

# Performs label encoding on given columns
def do_cat_le(X, X_test, cols):
    new_cols = []
    X_tr, X_te = pd.DataFrame(), pd.DataFrame()
    for c in cols:
        mask_tr = X[c].isnull()
        mask_te = X_test[c].isnull()
        c_name = c + '_le'
        new_cols.append(c_name)

        le = LabelEncoder()
        le.fit(list(X[c].astype(str).values) + list(X_test[c].astype(str).values))

        X_tr[c_name] = le.transform(list(X[c].astype(str).values)).where(~mask_tr)
        X_te[c_name] = le.transform(list(X_test[c].astype(str).values)).where(~mask_te)
    print(f'do_cat_le: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# Performs one hot encoding on given columns
def do_cat_ohe(X, X_test, cols):
    X_len = len(X)
    data = pd.concat([X, X_test])
    data = pd.get_dummies(data[cols])
    X_tr = data[:X_len]
    X_te = data[X_len:]
    new_cols = list(X_tr.columns)
    print(f'do_cat_ohe: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# Performs hash encoding on given columns
def do_cat_hash(X, X_test, cols):
    X_tr = pd.DataFrame()
    X_te = pd.DataFrame()
    new_cols = []
    for c in cols:
        c_name = c + '_hash'
        new_cols.append(c_name)
        size = X[c].nunique()
        X_tr[c_name] = X[c].apply(lambda x: hash(str(x)) % size)
        X_te[c_name] = X_test[c].apply(lambda x: hash(str(x)) % size)
    print(f'do_cat_hash: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# Performs bin encoding on given columns
def do_cat_bin(X, X_test, cols):
    be = BinaryEncoder(cols=cols).fit(X[cols])
    X_tr = be.transform(X[cols])
    X_te = be.transform(X_test[cols])
    new_cols = list(X_tr.columns)
    print(f'do_cat_bin: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# Performs frequency encoding on given columns
def do_cat_freq(X, X_test, cols):
    X_tr = pd.DataFrame()
    X_te = pd.DataFrame()
    new_cols = []
    for c in cols:
        c_name = c + '_freq'
        new_cols.append(c_name)
        tmp = pd.concat([X[[c]], X_test[[c]]])
        enc = tmp[c].value_counts().to_dict()
        X_tr[c_name] = X_tr[c].map(enc)
        X_te[c_name] = X_te[c].map(enc)
    print(f'do_cat_freq: Done. Added {len(new_cols)} new columns.')
    return X_tr, X_te, new_cols


# --------------------------------------------------------------------------------------------
# Numerical
# --------------------------------------------------------------------------------------------

# Performs standard scaling  on given columns
def do_num_standard(X, X_test, cols):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[cols])
    X_te = sc.transform(X_test[cols])
    return X_tr, X_te


# Performs minmax scaling  on given columns
def do_num_minmax(X, X_test, cols):
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(X[cols])
    X_te = sc.transform(X_test[cols])
    return X_tr, X_te


# --------------------------------------------------------------------------------------------
# Special
# --------------------------------------------------------------------------------------------

def get_feat_knn(df, cols, preffix='knn_'):
    scaler = StandardScaler()
    scaler.fit(df[cols])
    df = pd.DataFrame(scaler.transform(df[cols]), columns=df.columns)

    neigh = NearestNeighbors(n_jobs=-1)
    neigh.fit(df)
    dists, _ = neigh.kneighbors(df)

    df[preffix + 'mean_dist'] = dists.mean(axis=1)
    df[preffix + 'max_dist'] = dists.max(axis=1)
    df[preffix + 'min_dist'] = dists.min(axis=1)
    return [df[preffix + 'mean_dist', preffix + 'max_dist', preffix + 'min_dist'],
            [preffix + 'mean_dist', preffix + 'max_dist', preffix + 'min_dist']]





# --------------------------------------------------------------------------------------------
class DoubleValidationEncoder:
    def __init__(self, cols, encoder, splits):
        self.cols = cols
        self.encoder = encoder
        self.encoders_dict = {}
        self.splits = splits

    def fit_transform(self, X, y: np.array):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for n_fold, (tr, val) in enumerate(self.splits):
            X_train, X_val = X.loc[tr].reset_index(drop=True), X.loc[val].reset_index(drop=True)
            y_train, y_val = y[tr], y[val]
            _ = self.encoder.fit_transform(X_train[self.cols], y_train)

            val_t = self.encoder.transform(X_val)
            val_t = val_t.fillna(np.mean(y_train))
            if n_fold == 0:
                cols_representation = np.zeros((X.shape[0], val_t.shape[1]))

            self.encoders_dict[n_fold] = self.encoder
            cols_representation[val, :] += val_t.values
        cols_representation = pd.DataFrame(cols_representation, columns=X.columns)
        return cols_representation

    def transform(self, X):
        X = X.reset_index(drop=True)
        cols_representation = None
        for encoder in self.encoders_dict.values():
            test_tr = encoder.transform(X)

            if cols_representation is None:
                cols_representation = np.zeros(test_tr.shape)
            cols_representation = cols_representation + test_tr / self.folds.n_splits
        cols_representation = pd.DataFrame(cols_representation, columns=X.columns)
        return cols_representation
