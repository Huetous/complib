import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------------
def fillna_median(df, col):
    return df[col].fillna(df[col].median(), inplace=True)


def fillna_mode(df, col):
    return df[col].fillna(df[col].mode()[0], inplace=True)


def fillna_norm(df, col):
    c_avg = df[col].mean()
    c_std = df[col].std()
    c_null_count = df[col].isnull().sum()

    c_null_random_list = np.random.randint(c_avg - c_std, c_avg + c_std, size=c_null_count)
    df[col][np.isnan(df[col])] = c_null_random_list
    return df[col]


def clip_outliers(df, col):
    q75, q25 = np.percentile(df[col], [75, 25])
    upper_whisker = q75 + (q75 - q25) * 1.5
    return df[col].clip(upper=upper_whisker)


# ------------------------------------------------------------------------------------------------------
def reduce_memory_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))


# ------------------------------------------------------------------------------------------------------
def get_single_unique(df):
    cols = [col for col in df.columns if df[col].nunique() <= 1]
    if len(cols) > 0:
        print('There are', len(cols), 'columns with single unique values.')
        print(cols)
        return cols
    else:
        print('There is none of columns with single unique value.')
        return None


def get_duplicates(df):
    duplicates = df[df.duplicated()]
    if len(duplicates) > 0:
        print('There are', len(duplicates), 'duplicates.')
        print(duplicates)
        return duplicates
    else:
        print('There is none of duplicates!')
        return None


def get_cols_with_null(df, threshold=0.99):
    percents = df.isnull().sum() / df.isnull().count()
    cols = [index for index in percents.index if percents[index] >= threshold]
    if len(cols) > 0:
        print('There are', len(cols), 'columns with', threshold, 'percent of null values.')
        print(cols)
        return cols
    else:
        print('There are none columns with', threshold, 'percent of null values.')
        return None


# ------------------------------------------------------------------------------------------------------
def show_cols_with_null(df):
    percents = df.isnull().sum() / df.isnull().count()
    print('Missing data')
    for index in percents.index:
        if percents[index] != 0:
            print('{} : {:.5f}%'.format(index, percents[index]))


def show_skewed(df):
    sk_df = pd.DataFrame([{'column': col, 'uniq': df[col].nunique(),
                           'skewness': df[col].value_counts(normalize=True).values[0] * 100} for col in df.columns])
    sk_df = sk_df.sort_values(['skewness'], ascending=False)
    print('Skewed features:')
    print(sk_df)
