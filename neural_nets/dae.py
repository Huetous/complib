from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Concatenate
from tensorflow.python.keras.optimizers import Adam, Nadam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.regularizers import l2
import numpy as np
import tensorflow as tf


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5

    return tf.py_func(fallback_auc, (y_true, y_pred), tf.double)


def get_callbacks():
    es = EarlyStopping(monitor='val_auc',
                       min_delta=0.001,
                       patience=2,
                       verbose=1,
                       mode='max',
                       baseline=None,
                       restore_best_weights=True)

    rlr = ReduceLROnPlateau(monitor='val_auc',
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6,
                            mode='max',
                            verbose=1)
    return [es, rlr]


class DAE:
    def get_dae(self, X):
        inp = Input((X.shape[1],))
        x = Dense(512, activation='relu')(inp)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(.2)(x)
        y = Dense(X.shape[1], activation='linear')(x)

        model = Model(inputs=inp, outputs=y)
        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def add_noise(self, arr, p):
        n, m = arr.shape
        idx_arr = range(n)
        swap_n = round(n * p)
        for i in range(m):
            col_vals = np.random.permutation(arr.iloc[:, i])
            swap_idx = np.random.choice(idx_arr, size=swap_n)
            arr.iloc[swap_idx, i] = np.random.choice(col_vals, size=swap_n)
        return arr

    def data_gen(self, X, swap_rate, batch_size):
        idxs = np.arange(X.shape[0])
        while True:
            np.random.shuffle(idxs)
            X_orig = X.iloc[idxs[:batch_size]]
            X_noisy = self.add_noise(X_orig, swap_rate)
            yield X_noisy, X_orig

    def fit_dae(self, X, batch_size=2 ** 11, epochs=1):
        gen = self.data_gen(X, .15, batch_size)
        dae = self.get_dae(X)
        dae.fit_generator(generator=gen,
                          steps_per_epoch=X.shape[0] // batch_size,
                          epochs=epochs,
                          use_multiprocessing=True,
                          verbose=1)
        dae.trainable = False
        dae.compile(optimizer='adam',
                    loss='mse')
        dae.summary()
        self.dae = dae

    def get_nn(self, dae):
        x1 = self.dae.layers[1].output
        x2 = self.dae.layers[2].output
        x3 = self.dae.layers[3].output
        x = Concatenate()([x1, x2, x3])

        x = Dense(500, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(0.2)(x)
        y = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=self.dae.input, outputs=y)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[auc])

        return model
