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

    def show_nunique(self, drop_single=True):
        unique_counts = self.df.nunique()
        unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})

        if drop_single:
            to_drop = list(unique_counts.index)
            self.df = self.df.drop(to_drop, axis=1)

        unique_stats.plot.hist(edgecolor='k', figsize=(7, 5))
        plt.ylabel('Frequency', size=14)
        plt.xlabel('Unique Values', size=14)
        plt.show()
        self.verbose('show_single_unique')

    def do_find_colinear(self, threshold=0.99, do_ohe=False, drop=False):
        if do_ohe:
            corr_matrix = pd.get_dummies(self.df).corr()
        else:
            corr_matrix = self.df.corr()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]

        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])
        for column in to_drop:
            corr_features = list(upper.index[upper[column].abs() > threshold])
            corr_values = list(upper[column][upper[column].abs() > threshold])
            drop_features = [column for _ in range(len(corr_features))]

            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        if drop:
            self.df = self.df.drop(to_drop, axis=1)

        corr_matrix_plot = corr_matrix.loc[list(set(record_collinear['corr_feature'])),
                                           list(set(record_collinear['drop_feature']))]

        f, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
                    linewidths=.25, cbar_kws={"shrink": 0.6})

        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]))
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]))
        plt.show()

        self.verbose('do_find_colinear')


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
