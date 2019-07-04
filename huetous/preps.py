from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, FeatureHasher
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# General (le,ohe,freq encoding, bining, projection num on categ, transformation num)
# Regression/NN (scaling)
# Trees
# Special (time-series, images, FM/FFM)
# Text (Vectorizers, TF-IDF, Embeddings)


def do_cycle(df, cols, preffix, transform, func_name):
    new_df = df.copy(deep=True)
    new_cols = []

    len_before = len(new_df.columns.values)
    for col in cols:
        new_df[preffix + col] = transform(new_df[col])
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


# --------------------------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------------------------
def do_cat_le(df, cols, preffix='_le_', params=None):
    le_enc = LabelEncoder()
    return do_cycle(df, cols, preffix,
                    le_enc.fit_transform, 'do_le')


def do_cat_ohe(df, cols, preffix='_ohe_', params=None):
    oh_enc = OneHotEncoder(**params, handle_unknown='ignore', sparse=False)
    return do_cycle(df, cols, preffix,
                    oh_enc.fit_transform, 'do_ohe')


def do_cat_freq(df, cols, preffix='_freq_', params=None):
    return 0


# --------------------------------------------------------------------------------------------
# Numerical
# --------------------------------------------------------------------------------------------
def do_num_st_scale(df, cols, preffix='_st_scale_', params=None):
    st_scaler = StandardScaler()
    return do_cycle(df, cols, preffix,
                    st_scaler.fit_transform, 'do_st_scale')


# --------------------------------------------------------------------------------------------
# Special
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------
def do_text_tf_idf(df, cols, preffix='_tfidf_', params=None):
    tfidf_vect = TfidfVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    tfidf_vect.fit_transform, 'do_tf_idf')


def do_text_cnt_vect(df, cols, preffix='_cnt_vect_', params=None):
    count_vect = CountVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    count_vect.fit_transform, 'do_cnt_vect')


def do_text_w2vec(df, cols, preffix='_w2vec_', params=None):
    return False


def do_text_hash_vect(df, cols, preffix='_hash_vect_', params=None):
    hash_vect = HashingVectorizer(**params)
    return do_cycle(df, cols, preffix,
                    hash_vect.fit_transform, 'do_hash_vect')


def do_text_feat_hash(df, cols, preffix='_feat_hash_', params=None):
    return False


def do_text_tokenize(df, cols, preffix='_token_', params=None):
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
