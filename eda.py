import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from scipy.stats import pearsonr
import pandas as pd


# Statistic info about dataset
# Plots
# Categories variense

# --------------------------------------------------------------------------------------------
def plot_oof_preds(target, preds, class_name):
    plot_data = pd.DataFrame()
    plot_data['pred'] = preds['pred']
    plot_data['target'] = target

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='pred', y='target',
                    data=plot_data.loc[plot_data[class_name] == class_name, ['pred', 'target']])
    plt.xlabel('pred')
    plt.ylabel('target')
    plt.title(f'{class_name}', fontsize=18)
    plt.show()


# --------------------------------------------------------------------------------------------
def show_corr_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
def show_():
    tmp = pd.crosstab(train['ProductCD'], train['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)
    total = len(train)
    plt.figure(figsize=(14, 10))
    plt.suptitle('ProductCD Distributions', fontsize=22)

    plt.subplot(221)
    g = sns.countplot(x='ProductCD', data=train)
    g.set_title("ProductCD Distribution", fontsize=19)
    g.set_xlabel("ProductCD Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    g.set_ylim(0, 500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x() + p.get_width() / 2.,
               height + 3,
               '{:1.2f}%'.format(height / total * 100),
               ha="center", fontsize=14)

    plt.subplot(222)
    g1 = sns.countplot(x='ProductCD', hue='isFraud', data=train)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x='ProductCD', y='Fraud', data=tmp, color='black', order=['W', 'H', "C", "S", "R"], legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title("Product CD by Target(isFraud)", fontsize=19)
    g1.set_xlabel("ProductCD Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)

    plt.subplot(212)
    g3 = sns.boxenplot(x='ProductCD', y='TransactionAmt', hue='isFraud',
                       data=train[train['TransactionAmt'] <= 2000])
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)

    plt.subplots_adjust(hspace=0.6, top=0.85)

    plt.show()

def show__():
    plt.figure(figsize=(14, 10))

    plt.subplot(221)
    g2 = sns.boxplot(train.TransactionAmt)
    g2.set_title('Box Plot of TRAIN Transaction Amount', fontsize=16)
    g2.set_xlabel('Values of TRAIN Transaction Amount', fontsize=16)

    plt.subplot(222)
    g1 = sns.boxplot(test.TransactionAmt)
    g1.set_title('Box Plot of TEST Transaction Amount', fontsize=20)
    g1.set_xlabel('Values of TEST Transaction Amount', fontsize=16)

    plt.subplot(212)
    g3 = sns.distplot(test.TransactionAmt, label='TEST')
    g3 = sns.distplot(train.TransactionAmt, label='TRAIN')
    g3.legend()
    g3.set_title('Distribution of Transaction Amount', fontsize=16)
    g3.set_xlabel('Values of Transaction Amount', fontsize=16)

    plt.subplots_adjust(hspace=0.9, top=0.8)


def plot_cat(dd, col):
    mean_fraud = dd["isFraud"].mean()

    plt.figure(figsize=(15, 10))

    max_show_cats = 30
    cnts = dd[col].value_counts(normalize=True, dropna=False)
    plot_cnts = cnts.iloc[:max_show_cats]
    plt.bar(range(len(plot_cnts)), plot_cnts, width=0.9, tick_label=plot_cnts.index)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Category rate")
    plt.grid(False)

    test_cnts = test[col].value_counts(normalize=True)
    plt.step(range(len(plot_cnts)), test_cnts.reindex(plot_cnts.index).fillna(0), c="b", where="mid", label="Test data")
    plt.legend()

    fraud_rate = dd.groupby(col)["isFraud"].mean()
    if dd[col].isnull().any():
        fraud_rate[np.nan] = dd.loc[dd[col].isnull(), "isFraud"].mean()

    ax_fraud = plt.gca().twinx()
    ax_fraud.stem(range(len(plot_cnts)), fraud_rate.reindex(plot_cnts.index), linefmt="k:", markerfmt="ko")
    ax_fraud.set_ylabel("Fraud rate", color="k")
    ax_fraud.grid(False)
    ax_fraud.axhline(mean_fraud, ls="--", c="k")
    ax_fraud.set_ylim(bottom=0)

    title = [col]
    if len(cnts) > len(plot_cnts):
        title.append(f"({plot_cnts.sum():.0%} of data; {len(plot_cnts)}/{len(cnts)} cats)")
    plt.title(" ".join(title))
    plt.show()