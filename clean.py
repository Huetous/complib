import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats


# ------------------------------------------------------------------------------------------------------
def fillna_median(df, c):
    return df[c].fillna(df[c].median(), inplace=True)


def fillna_mode(df, c):
    return df[c].fillna(df[c].mode()[0], inplace=True)


def fillna_norm(df, c):
    c_avg = df[c].mean()
    c_std = df[c].std()
    cnt = df[c].isnull().sum()

    rand_nums = np.random.randint(c_avg - c_std, c_avg + c_std, size=cnt)
    df[c][np.isnan(df[c])] = rand_nums
    return df[c]


def clip_outliers(df, c, qs=None):
    if qs is None:
        qs = [1, 99]
    q1, q2 = np.percentile(df[c], [qs[0], qs[1]])
    upper_whisker = q1 + (q2 - q1) * 1.5
    return df[c].clip(upper=upper_whisker)


# ------------------------------------------------------------------------------------------------------
def reduce_memory_usage(df):
    types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for c in tqdm(df.columns):
        c_type = df[c].dtypes
        if c_type in types:
            c_min = df[c].min()
            c_max = df[c].max()
            if str(c_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[c] = df[c].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[c] = df[c].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[c] = df[c].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[c] = df[c].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[c] = df[c].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[c] = df[c].astype(np.float32)
                else:
                    df[c] = df[c].astype(np.float64)
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


def get_summary(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values / df.isnull().count()
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2)
    summary['Skew'] = stats.skew(df)

    return summary


# ------------------------------------------------------------------------------------------------------


def show_cols_with_null(df):
    percents = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    print('Missing data')
    for index in percents.index:
        if percents[index] != 0:
            print('{} : {:.5f}'.format(index, percents[index]))


def show_skewed(df):
    sk_df = pd.DataFrame([{'column': col, 'uniq': df[col].nunique(),
                           'skewness': stats.skew(df[col])} for col in df.columns])
    sk_df = sk_df.sort_values(['skewness'], ascending=False)
    print('Skewed features:')
    print(sk_df)


# -------------------------------------------------------------------------------------------------------

