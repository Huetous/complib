import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------------------------------------------
def do_fillna_median(df, col, astype=np.int16):
    return df[col].fillna(df[col].median(), inplace=True).astype(astype)


def do_fillna_mode(df, col, astype=np.int16):
    return df[col].fillna(df[col].mode()[0], inplace=True).astype(astype)


def do_clip_outliers(df, col):
    q75, q25 = np.percentile(df[col], [75, 25])
    iqr = q75 - q25
    upper_whisker = q75 + iqr * 1.5
    return df[col].clip(upper=upper_whisker)


def do_fillna_meanstd(df, col, astype=np.int16):
    age_avg = df[col].mean()
    age_std = df[col].std()
    age_null_count = df[col].isnull().sum()
    age_null_random_list = \
        np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df[col][np.isnan(df[col])] = age_null_random_list
    return df[col].astype(astype)


def do_mapping(df, col, map_, astype=np.int16):
    return df[col].map(map_).astype(astype)


# ------------------------------------------------------------------------------------------------------
def do_reduce_memory_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
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
# def do_find_collinear(df, threshold=0.99, do_ohe=False, drop=False):
#     if do_ohe:
#         corr_matrix = pd.get_dummies(df).corr()
#     else:
#         corr_matrix = df.corr()
#
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#     to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
#
#     record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])
#     for column in to_drop:
#         corr_features = list(upper.index[upper[column].abs() > threshold])
#         corr_values = list(upper[column][upper[column].abs() > threshold])
#         drop_features = [column for _ in range(len(corr_features))]
#
#         temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
#                                           'corr_feature': corr_features,
#                                           'corr_value': corr_values})
#         record_collinear = record_collinear.append(temp_df, ignore_index=True)
#
#     corr_matrix_plot = corr_matrix.loc[list(set(record_collinear['corr_feature'])),
#                                        list(set(record_collinear['drop_feature']))]
#
#     print(record_collinear)
#     #
#     # f, ax = plt.subplots(figsize=(10, 8))
#     # cmap = sns.diverging_palette(220, 10, as_cmap=True)
#     # _ = sns.heatmap(
#     #     corr_matrix_plot,
#     #     cmap=cmap,
#     #     center=0,
#     #     square=True,
#     #     cbar_kws={'shrink': .9},
#     #     ax=ax,
#     #     annot=True,
#     #     linewidths=0.1, vmax=1.0, linecolor='white',
#     #     annot_kws={'fontsize': 12}
#     # )
#     #
#     # ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
#     # ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]))
#     # ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
#     # ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]))
#     # plt.show()
#     if drop:
#         df = df.drop(to_drop, axis=1)
#         return df
#

def do_drop_single_unique(df):
    unique_counts = df.nunique()
    unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'}).sort_values(
        ascending=True)

    to_drop = []
    for feat in unique_stats:
        if feat['nunique'] == 1:
            to_drop.append(feat['feature'])
    return df.drop(to_drop, axis=1)


# ------------------------------------------------------------------------------------------------------
def show_nunique(df):
    unique_counts = df.nunique()
    unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'}). \
        sort_values(['nunique'], ascending=True)
    print(unique_stats)


def show_missing_data(df, plot=False):
    feats_name = []
    feats_percent = []
    percents = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)

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


def show_duplicates(df):
    if len(df[df.duplicated()]) > 0:
        print('Number of duplicates:', len(df[duplicates]))
        print(df[duplicates])
    else:
        print('There is none of duplicates!')


def show_skewed(df):
    sk_df = pd.DataFrame([{'column': col, 'uniq': df[col].nunique(),
                           'skewness': df[col].value_counts(normalize=True).values[0] * 100} for col in
                          df.columns])
    sk_df = sk_df.sort_values(['skewness'], ascending=False)
    print('Skewed features: \n')
    print(sk_df)
