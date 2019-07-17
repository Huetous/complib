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
    ensembled_data = pd.DataFrame()
    cols = []
    for filename in pickle_list:
        data, cols = load_pickle(filename)
        ensembled_data = np.concatenate([ensembled_data, data], axis=1)
        cols.append(cols)
    return ensembled_data, np.array(cols).reshape(-1,)



