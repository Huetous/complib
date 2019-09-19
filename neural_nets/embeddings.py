from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding, Concatenate
from tensorflow.python.keras.layers import SpatialDropout1D, BatchNormalization
from tensorflow.python.keras.layers import Input
import numpy as np

def create_model_emb(df):
    nunique = df.nunique().sum()
    emb_size = int(min(50, nunique // 2))
    max_len = df.shape[1]

    model = Sequential()
    model.add(Embedding(nunique, emb_size, input_length=max_len, name="embedding"))

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(2 ** 6, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2 ** 6, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model


def create_model_N_emb(df):
    inps = []
    outs = []
    for c in df:
        nunique = np.max(c)
        emb_size = int(min(50, nunique // 2))
        inp = Input(shape=(1,))
        out = Embedding(nunique + 1, emb_size, input_length=1)(inp)
        out = SpatialDropout1D(0.3)(out)
        inps.append(inp)
        outs.append(out)

    x = Concatenate()(outs)
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(2 ** 8, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    x = Dense(2 ** 8, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inps, outputs=y)
    return model
