import pickle
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
def save_pickle(df, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        return [data, list(data.columns.values)]


def ensemble_data(pickle_list):
    ensembled_data, ensembled_cols = load_pickle(pickle_list[0])
    for filename in pickle_list:
        if filename == pickle_list[0]:
            continue
        data, cols = load_pickle(filename)
        ensembled_data = np.hstack((ensembled_data, data))
        for col in cols:
            ensembled_cols.append(col)
    return pd.DataFrame(ensembled_data, columns=ensembled_cols), ensembled_cols
