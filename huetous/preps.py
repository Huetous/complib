from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, FeatureHasher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from multiprocessing import Pool


# General (le,ohe,freq encoding, bining, projection num on categ, transformation num)
# Regression/NN (scaling)
# Trees
# Special (time-series, images, FM/FFM)
# Text (Vectorizers, TF-IDF, Embeddings)


def do_cycle_memorysafe(df, cols, preffix, transform, params, func_name):
    new_df = df[cols].copy(deep=True)
    new_cols = []

    for col in cols:
        new_df[preffix + col] = transform(new_df[col], **params)
        new_cols.append(preffix + col)

    print(func_name + ': Done')
    print('Added ', new_df.shape[1], ' new columns.')
    return [new_df, new_cols]


def do_cycle(df, cols, preffix, transform, params, func_name):
    new_df = df.copy(deep=True)
    new_cols = []

    len_before = len(new_df.columns.values)
    for col in cols:
        new_df[preffix + col] = transform(new_df[col], **params)
        new_cols.append(preffix + col)
    len_after = len(new_df.columns.values)

    print(func_name + ': Done')
    print('Added ', (len_before - len_after), ' new columns.')
    return [new_df, new_cols]


# --------------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------------
def do_date_extract(df, col, preffix='date_', params=None):
    new_df = df.copy(deep=True)
    new_cols = []
    date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']

    new_df[col] = pd.to_datetime(df[col])

    len_before = len(new_df.columns.values)
    for part in date_parts:
        new_df[col + preffix + part] = getattr(new_df[col].dt, part).astype(int)
        new_cols.append(col + preffix + part)
    len_after = len(new_df.columns.values)

    print('do_date_extract: Done')
    print('Added ', (len_before - len_after), ' new columns.')
    return [new_df, new_cols]


def get_trend_feat(df, col, preffix='trend_', params=None):
    idx = np.array(range(len(df[col])))
    new_df = df.copy(deep=True)

    if params['abs'] is True:
        new_df[col] = np.abs(new_df[col])

    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), new_df[col])
    new_df[preffix + col] = lr.coef_[0]
    return [new_df, preffix + col]


# --------------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------------
def do_cat_le(df, cols, preffix='le_', params=None):
    le_enc = LabelEncoder()
    return do_cycle(df, cols, preffix,
                    le_enc.fit_transform, None, 'do_cat_le')


def do_cat_le_tr_te(train, test, cols, preffix='train_test_le_', params=None):
    le = LabelEncoder()
    new_train = train.copy(deep=True)
    new_test = test.copy(deep=True)
    new_cols = []
    for col in cols:
        le.fit(new_train[col].append(new_test[col]))
        new_cols.append(preffix + col)
        new_train[preffix + col] = le.transform(new_train[col])
        new_test[preffix + col] = le.transform(new_test[col])
    return [new_train, new_test, ]


def do_cat_target_expand_mean(df, cols, target, preffix='target_expand_mean_'):
    new_df = df.copy(deep=True)
    new_cols = []

    for col in cols:
        cumsum = new_df.groupby(col)[target].cumsum() - new_df[target]
        cumcnt = new_df.groupby(col).cumcount()
        new_df[preffix + col] = cumsum / cumcnt
        new_df[preffix + col].fillna(0.3343, inplace=True)
        new_cols.append(preffix + col)
    return [new_df, new_cols]



# -----------------------------------------------------
# Target Mean Encoding
# -----------------------------------------------------
# def for_target_enc(col, train, test, preffix, skf):
#     train[preffix + col] = 0
#
#     for i, (tr_idx, te_idx) in enumerate(skf.split(train, train.TARGET)):
#         enc = train.iloc[tr_idx].groupby(col)['TARGET'].mean()
#         train.set_index(col, inplace=True)
#         train.iloc[te_idx, -1] = enc
#         train.reset_index(inplace=True)
#
#     enc = train.groupby(col)['TARGET'].mean()
#     test[preffix + col] = 0
#     test.set_index(col, inplace=True)
#     test.iloc[:, -1] = enc
#     test.reset_index(inplace=True)
#
#
# def do_cat_target_enc(train, test, cols, preffix='target_mean_', params=None):
#     params['n_splits'] = 5
#     params['random_state'] = 42
#     params['shuffle'] = True
#     skf = StratifiedKFold(**params)
#
#     new_train = train[cols].copy(deep=True)
#     new_test = test[cols].copy(deep=True)
#     new_cols = []
#     for col in cols:
#         for_target_enc(col, new_train, new_test, preffix, skf)
#         new_cols.append(preffix + col)
#     return [new_train, new_test, new_cols]
#
#
# def do_cat_target_enc_multi(train, test, cols, preffix='target_mean_multi_', params=None):
#     cat_comb = list(combinations(cols, 2))
#     new_train = train[cols].copy(deep=True)
#     new_test = test[cols].copy(deep=True)
#     comb_cols = []
#
#     for c1, c2 in cat_comb:
#         new_train[preffix + f'{c1}-{c2}'] = new_train[c1] + new_train[c2]
#         new_test[preffix + f'{c1}-{c2}'] = new_test[c1] + new_test[c2]
#         comb_cols.append(preffix + f'{c1}-{c2}')
#
#     pool = Pool(10)
#     pool.map(for_target_enc, comb_cols)
#     pool.close()
#     return [new_train, new_test, comb_cols]
#

# -----------------------------------------------------

def do_cat_ohe(df, cols, preffix='ohe_', params=None):
    oh_enc = OneHotEncoder(**params, handle_unknown='ignore', sparse=False)
    return do_cycle(df, cols, preffix,
                    oh_enc.fit_transform, None, 'do_cat_ohe')


def do_cat_dummmy(df, cols, preffix='dummy_', params=None):
    params['drop_first'] = True
    return do_cycle(df, cols, preffix,
                    pd.get_dummies, params, 'do_cat_dummy')


def do_cat_freq(df, cols, preffix='_freq_', params=None):
    return 0


def do_cat_dummy(df, cols, preffix='dummy_', params=None):
    params['drop_first'] = True
    return do_cycle(df, cols, preffix,
                    pd.get_dummies, params, 'do_cat_dummy')


# --------------------------------------------------------------------------------------------
# Numerical
# --------------------------------------------------------------------------------------------
def do_num_st_scale(df, cols, preffix='st_scale_', params=None):
    st_scaler = StandardScaler()
    return do_cycle(df, cols, preffix,
                    st_scaler.fit_transform, None, 'do_st_scale')


def do_num_minmax_scale(df, cols, preffix='minmax_', params=None):
    minmax_scaler = MinMaxScaler()
    return do_cycle(df, cols, preffix,
                    minmax_scaler.fit_transform, None, 'do_num_minmax_scale')


def do_num_cut(df, cols, preffix='cut_', params=None):
    return do_cycle(df, cols, preffix,
                    pd.cut, None, 'do_num_cut')


def do_num_qcut(df, cols, preffix='cut_', params=None):
    return do_cycle(df, cols, preffix,
                    pd.qcut, None, 'do_num_qcut')


# --------------------------------------------------------------------------------------------
# Special
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------
def do_text_tf_idf(df, cols, preffix='_tfidf_', params=None):
    tfidf_vect = TfidfVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    tfidf_vect.fit_transform, None, 'do_tf_idf')


def do_text_cnt_vect(df, cols, preffix='_cnt_vect_', params=None):
    count_vect = CountVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    count_vect.fit_transform, None, 'do_cnt_vect')


def do_text_w2vec(df, cols, preffix='_w2vec_', params=None):
    return False


def do_text_hash_vect(df, cols, preffix='_hash_vect_', params=None):
    hash_vect = HashingVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    hash_vect.fit_transform, None, 'do_hash_vect')


def do_text_feat_hash(df, cols, preffix='_feat_hash_', params=None):
    return False


def do_text_tokenize(df, cols, preffix='_token_', params=None):
    tokenz = Tokenizer(**params)
    return do_cycle(df, cols, preffix,
                    tokenz.sequences_to_matrix, None, 'do_tokenize')


# --------------------------------------------------------
def do_text_remove_stopwords(data):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(data)
    words_filtered = []
    for w in words:
        if w not in stop_words:
            words_filtered.append(w)
    print('do_remove_stopwords: Done')
    print('Words extracted: ', len(words_filtered))
    return words_filtered


# --------------------------------------------------------------------------------------------

def get_feat_knn(df, cols, preffix='knn_', params=None):
    new_df = df.copy(deep=True)
    scaler = StandardScaler()
    scaler.fit(new_df[cols])
    new_df = pd.DataFrame(scaler.transform(new_df[cols]), columns=new_df.columns)

    neigh = NearestNeighbors(**params, n_jobs=-1)
    neigh.fit(new_df)
    dists, _ = neigh.kneighbors(new_df, **params)

    new_df[preffix + 'mean_dist'] = dists.mean(axis=1)
    new_df[preffix + 'max_dist'] = dists.max(axis=1)
    new_df[preffix + 'min_dist'] = dists.min(axis=1)
    return [new_df, [preffix + 'mean_dist', preffix + 'max_dist', preffix + 'min_dist']]


# --------------------------------------------------------------------------------------------
# Times-series and signal processing
def get_rolling_mean(df, cols, preffix='rolling_', parans=None):
    return [0, 0]


# //////////////////////////////////////
#           Neural Network
# //////////////////////////////////////
def datagen_augment(datadir, target_size=(150, 150), batch_size=32, class_mode='binary'):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    data_generator = datagen.flow_from_directory(
        datadir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
    )
    return data_generator


def feat_ext(model, dir, datagen, sample_count, batch_size=20, target_size=(150, 150), class_mode='binary'):
    feats = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,))
    gen = datagen.flow_from_directory(
        dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
    )
    i = 0
    for inputs_batch, labels_batch in gen:
        feats_batch = model.predict(inputs_batch)
        feats[i * batch_size: (i + 1) * batch_size] = feats_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return feats, labels

# Feature extraction. conv_base not changes
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Dense(256...))
# conv_base.trainable = False

# Fine-tuning. conv_base top layers are unfreezed. They will retrained
# for new task
