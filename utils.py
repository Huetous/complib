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

# Assembles data from pickles in given list
def assemble_data(pickle_list):
    assembled_data, assembled_cols = load_pickle(pickle_list[0])
    for filename in pickle_list:
        if filename == pickle_list[0]:
            continue
        data, cols = load_pickle(filename)
        assembled_data = np.hstack((assembled_data, data))
        for col in cols:
            assembled_cols.append(col)
    return pd.DataFrame(assembled_data, columns=assembled_cols), assembled_cols
