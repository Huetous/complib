import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Performs distribution smoothing - takes only those values in each column that appear at train and test often
def distribution_smoothing(train, test, cols):
    for col in cols:
        print('before. nunique {} Train {}, Test {}'.format(col, train[col].nunique(), test[col].nunique()))

        agg_tr = train.groupby([col]).aggregate({col: 'count'}).rename(columns={col: 'Train'}).reset_index()
        agg_te = test.groupby([col]).aggregate({col: 'count'}).rename(columns={col: 'Test'}).reset_index()
        agg = pd.merge(agg_tr, agg_te, on=col, how='outer')

        agg['Total'] = agg['Train'] + agg['Test']
        agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]
        agg[col + '_Copy'] = agg[col]

        train[col] = pd.merge(train[[col]], agg[[col, col + '_Copy']], on=col, how='left')[col + '_Copy']
        test[col] = pd.merge(test[[col]], agg[[col, col + '_Copy']], on=col, how='left')[col + '_Copy']

        print('after.  nunique {} Train {}, Test {}'.format(col, train[col].nunique(), test[col].nunique()))
        print('-' * 20)


# Performs Kolmogorov-Smitnov test on features distributions
def do_ks_2samp(train, test):
    rej = []
    not_rej = []

    for col in train.columns:
        statistic, pvalue = ks_2samp(train[col], test[col])
        if pvalue >= statistic:
            not_rej.append(col)
        if pvalue < statistic:
            rej.append(col)

        plt.figure(figsize=(8, 4))
        plt.title("Kolmogorov-Smirnov test for train/test\n"
                  "feature: {}, statistics: {:.5f}, pvalue: {:5f}".format(col, statistic, pvalue))
        sns.kdeplot(train[col], color='blue', shade=True, label='Train')
        sns.kdeplot(test[col], color='green', shade=True, label='Test')

        plt.show()
    return rej, not_rej
