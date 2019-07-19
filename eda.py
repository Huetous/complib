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
def show_pairplot(df):
    sns.pairplot(df)
    plt.show()

