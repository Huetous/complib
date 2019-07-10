from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Categories cleaning
# Categories concatenation

class Cleaner():
    def __init__(self, df):
        self.df = df.copy(deep=True)

    def return_df(self):
        self.verbose('return_df')
        return self.df

    def verbose(self, string):
        print(string + ': Done')

    def do_fillna_median(self, col, astype=np.int16):
        self.df[col].fillna(self.df[col].median(), inplace=True).astype(astype)
        self.verbose('do_fillna_median')

    def do_fillna_mode(self, col, astype=np.int16):
        self.df[col].fillna(self.df[col].mode()[0], inplace=True).astype(astype)
        self.verbose('do_fillna_mode')

    def do_clip_outliers(self, col):
        q75, q25 = np.percentile(self.df[col], [75, 25])
        iqr = q75 - q25
        upper_whisker = q75 + iqr * 1.5
        self.df[col] = self.df[col].clip(upper=upper_whisker)
        self.verbose('do_clip_outliers')

    def do_fillna_meanstd(self, col, astype=np.int16):
        age_avg = self.df[col].mean()
        age_std = self.df[col].std()
        age_null_count = self.df[col].isnull().sum()
        age_null_random_list = \
            np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        self.df[col][np.isnan(self.df[col])] = age_null_random_list
        self.df[col] = self.df[col].astype(astype)
        self.verbose('do_fillna_meanstd')

    def do_mapping(self, col, map, astype=np.int16):
        self.df[col] = self.df[col].map(map).astype(astype)
        self.verbose('do_mapping')

    def do_reduce_memory_usage(self, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.df.memory_usage().sum() / 1024 ** 2
        for col in self.df.columns:
            col_type = self.df[col].dtypes
            if col_type in numerics:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
        end_mem = self.df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))

    def do_was_missing_cols(self):
        cols_with_missing = [col for col in self.df.columns if self.df[col].isnull().any()]
        for col in cols_with_missing:
            self.df[col + "_was_missing"] = self.df[col].isnull()
        self.verbose('do_was_missing_cols')

    def show_missing_data(self, plot=False):
        feats_name = []
        feats_percent = []
        percents = (self.df.isnull().sum() / self.df.isnull().count()).sort_values(ascending=False)

        print('Missing data')
        for index in percents.index:
            if percents[index] != 0:
                feats_name.append(index)
                feats_percent.append(percents[index])
                print('{} : {:.5f}'.format(index, percents[index]))
        print("\n")
        if plot:
            f, ax = plt.subplots(figsize=(9, 6))
            plt.xticks(rotation='90')
            sns.barplot(x=feats_name, y=feats_percent)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Percent missing data by feature', fontsize=15)
            plt.show()

    def show_duplicates(self):
        duplicates = self.df[self.df.duplicated()]
        print('Number of duplicates:', len(duplicates))
        print(duplicates)

    def show_skewed(self):
        sk_df = pd.DataFrame([{'column': col, 'uniq': self.df[col].nunique(),
                               'skewness': self.df[col].value_counts(normalize=True).values[0] * 100} for col in
                              self.df.columns])
        sk_df = sk_df.sort_values('skewness', ascending=False)
        print('Skewed features: \n')
        print(sk_df)
