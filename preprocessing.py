from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn import model_selection
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, FeatureHasher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from multiprocessing import Pool
from huelib import clean, mean_target_encoder


# General (le,ohe,freq encoding, bining, projection num on categ, transformation num)
# Regression/NN (scaling)
# Trees
# Special (time-series, images, FM/FFM)
# Text (Vectorizers, TF-IDF, Embeddings)


def do_cycle(df, cols, preffix, transform, func_name, params=None):
    new_cols = []
    for col in tqdm(cols):
        if params is None:
            df[preffix + col] = transform(df[col])
        else:
            df[preffix + col] = transform(df[col], **params)
        new_cols.append(preffix + col)

    print('Added', len(new_cols), 'new columns.')
    print(func_name + ': Done')
    return [df[new_cols], new_cols]


# --------------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------------
def do_date_extract(df, col, preffix='date_'):
    new_cols = []
    date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']

    df[col] = pd.to_datetime(df[col])
    for part in date_parts:
        df[col + preffix + part] = getattr(df[col].dt, part).astype(int)
        new_cols.append(col + preffix + part)

    print('do_date_extract: Done')
    print('Added', len(new_cols), 'new columns.')
    return [df[new_cols], new_cols]


def do_was_missing_cols(df, preffix='was_missing_'):
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    new_cols = []
    for col in cols_with_missing:
        df[preffix + col] = df[col].isnull()
        new_cols.append(preffix + col)

    print('do_was_missing_cols: Done')
    print('Added', len(new_cols), 'new columns.')
    return [df[new_cols], new_cols]


# --------------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------------
# def do_cat_le(df, cols, preffix='le_'):
#     return do_cycle(df, cols, preffix, LabelEncoder().fit_transform, 'do_cat_le')


def do_cat_le(train, test, cols, preffix='tr_te_le_'):
    le = LabelEncoder()
    new_cols = []
    for col in cols:
        le.fit(train[col].append(test[col]))
        train[preffix + col] = le.transform(train[col])
        test[preffix + col] = le.transform(test[col])

        new_cols.append(preffix + col)

    return [train[new_cols], test[new_cols], new_cols]


def do_cat_mte(train, test=None, cols=None, target_col=None,
               n_splits=3, shuffle=False, seed=0, alpha=5,
               preffix='mte_'):
    if cols is None:
        raise ValueError('Columns for encoding should be specified.')
    if target_col is None:
        raise ValueError('Target columns should be specified.')

    mte = mean_target_encoder.TargetEncoderCV(cols=cols, n_splits=n_splits,
                                              shuffle=shuffle, seed=seed, alpha=alpha)
    new_train = mte.fit_transform(train, train[target_col])

    if test is not None:
        new_test = mte.transform(test)
    else:
        new_test = None

    for col in cols:
        new_train = new_train.rename(columns={str(col): str(preffix + col)})
        if test is not None:
            new_test = new_test.rename(columns={str(col): str(preffix + col)})
    return [new_train, new_test, list(new_train.columns.values)]


def do_cat_dummy(df, cols, preffix='dummy_'):
    new_cols = []
    for col in cols:
        df[preffix + col] = pd.get_dummies(df[col], drop_first=True)
        new_cols.append(preffix + col)

    print('do_cat_dummy: Done')
    print('Added', len(new_cols), 'new columns.')
    return [df[new_cols], new_cols]


def do_cat_freq(df, cols, preffix='freq_', params=None):
    return 0


# --------------------------------------------------------------------------------------------
# Numerical
# --------------------------------------------------------------------------------------------
def do_num_z_scale(df, cols, preffix='z_scale_'):
    st_scaler = StandardScaler()
    new_cols = []
    for col in cols:
        df[preffix + col] = st_scaler.fit_transform(df[[col]])
        new_cols.append(preffix + col)

    clean.reduce_memory_usage(df[new_cols])
    return [df[new_cols], new_cols]


def do_num_minmax_scale(df, cols, preffix='minmax_scale_'):
    return do_cycle(df, cols, preffix, MinMaxScaler().fit_transform, 'do_num_minmax_scale')


def do_num_cut(df, cols, preffix='cut_', params=None):
    if params is None:
        raise Exception('Number of bins should be set.')
    return do_cycle(df, cols, preffix, pd.cut, 'do_num_cut')


def do_num_qcut(df, cols, preffix='qcut_', params=None):
    if params is None:
        raise Exception('Number of quantiles should be set.')
    return do_cycle(df, cols, preffix, pd.qcut, 'do_num_qcut', params)


# --------------------------------------------------------------------------------------------
# Special
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------
def do_text_tf_idf(df, cols, preffix='tfidf_', params=None):
    if params is not None:
        tfidf_vect = TfidfVectorizer(**params)
    else:
        tfidf_vect = TfidfVectorizer()
    return do_cycle(df, cols, preffix,
                    tfidf_vect.fit_transform, 'do_tf_idf')


def do_text_cnt_vect(df, cols, preffix='cnt_vect_', params=None):
    if params is not None:
        count_vect = CountVectorizer(**params)
    else:
        count_vect = CountVectorizer()
    return do_cycle(df, cols, preffix,
                    count_vect.fit_transform, 'do_cnt_vect')


def do_text_w2vec(df, cols, preffix='w2vec_', params=None):
    return False


def do_text_hash_vect(df, cols, preffix='hash_vect_', params=None):
    if params is not None:
        hash_vect = HashingVectorizer(**params)
    else:
        hash_vect = HashingVectorizer()
    return do_cycle(df, cols, preffix,
                    hash_vect.fit_transform, 'do_hash_vect')


def do_text_feat_hash(df, cols, preffix='feat_hash_', params=None):
    return False


def do_text_tokenize(df, cols, preffix='token_', params=None):
    tokenz = Tokenizer(**params)
    return do_cycle(df, cols, preffix,
                    tokenz.sequences_to_matrix, 'do_tokenize')


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
# Times-series and signal processing
#
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
