from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


class Cleaner():
    def __init__(self, df):
        self.df = df.copy(deep=True)

    def do_fillna_median(self, col, astype=np.int16):
        self.df[col].fillna(df[col].median(), inplace=True).astype(astype)

    def do_fillna_mode(self, col, astype=np.int16):
        self.df[col].fillna(df[col].mode()[0], inplace=True).astype(astype)

    def do_fillna_meanstd(self, col, astype=np.int16):
        age_avg = self.df[col].mean()
        age_std = self.df[col].std()
        age_null_count = self.df[col].isnull().sum()
        age_null_random_list = \
            np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        self.df[col][np.isnan(df[col])] = age_null_random_list
        self.df[col] = self.df[col].astype(vartype)

    def do_mapping(self, col, map, astype=np.int16):
        self.df[col] = self.df[col].map(map).astype(var_type)

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
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))

    def do_was_missing_cols(self):
        cols_with_missing = [col for col in self.df.columns if self.df[col].isnull().any()]
        for col in cols_with_missing:
            self.df[col + "_was_missing"] = self.df[col].isnull()

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

# Fillna
# Categories cleaning
# Categories concatenation



# # --------------------------------------------------------------------------------------------
# def do_fillna(dataset, column, type, vartype=int):
#     if type == 'median':
#         dataset[column].fillna(dataset[column].median(), inplace=True)
#     if type == 'mode':
#         dataset[column].fillna(dataset[column].mode()[0], inplace=True)
#     if type == 'meanstd':
#         age_avg = dataset[column].mean()
#         age_std = dataset[column].std()
#         age_null_count = dataset[column].isnull().sum()
#         age_null_random_list = \
#             np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
#         dataset[column][np.isnan(dataset[column])] = age_null_random_list
#         dataset[column] = dataset[column].astype(vartype)
#
#
# # --------------------------------------------------------------------------------------------
# def do_mapping(dataframe, col, map, var_type):
#     dataframe[col] = dataframe[col].map(map).astype(var_type)
#

# --------------------------------------------------------------------------------------------
# def do_reduce_memory_usage(df, verbose=True):
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#     end_mem = df.memory_usage().sum() / 1024 ** 2
#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
#             start_mem - end_mem) / start_mem))
#     return df


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Create new column which will tell what value was missed
# --------------------------------------------------------------------------------------------
# def do_was_missing_cols(df):
#     df_copy = df.copy()
#
#     cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
#     for col in cols_with_missing:
#         df_copy[col + "_was_missing"] = df_copy[col].isnull()
#
#     imp_df_copy = pd.DataFrame(SimpleImputer().fit_transform(df_copy))
#     imp_df_copy.columns = df_copy.columns
#     return imp_df_copy


# --------------------------------------------------------------------------------------------
# def show_missing_data(dataframe, plot=False):
#     feats_name = []
#     feats_percent = []
#     percents = (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending=False)
#
#     print('Missing data')
#     for index in percents.index:
#         if percents[index] != 0:
#             feats_name.append(index)
#             feats_percent.append(percents[index])
#             print('{} : {:.5f}'.format(index, percents[index]))
#     print("\n")
#     if plot:
#         f, ax = plt.subplots(figsize=(9, 6))
#         plt.xticks(rotation='90')
#         sns.barplot(x=feats_name, y=feats_percent)
#         plt.xlabel('Features', fontsize=15)
#         plt.ylabel('Percent of missing values', fontsize=15)
#         plt.title('Percent missing data by feature', fontsize=15)
#         plt.show()
#
#
# # --------------------------------------------------------------------------------------------
# def show_duplicates(df):
#     duplicates = df[df.duplicated()]
#     print('Number of duplicates:', len(duplicates))
#     print(duplicates)
