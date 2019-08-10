import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from data import BINARY_CLASS_SETS
from config import OUT_DIR, VIS_DIR, prepare_env


def result_path(name: str, out_dir=OUT_DIR):
    filename = f'{name.lower()}experiment-ensemble.csv'
    return os.path.join(out_dir, filename)


def scatterplot(df, name, x, y='accuracy', datasets=['iris', 'cancer']):
    ncols = 3
    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(ncols * 5, nrows * 5))

    for ax, dataset_name in zip(axes.flat, datasets):
        data_df = df[df['dataset_name'] == dataset_name]

        values = data_df[[x, y]].values
        xs, ys = values[:, 0], values[:, 1]

        ax.set_title(dataset_name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.scatter(xs, ys)

    fig.suptitle(name)
    fig.savefig(os.path.join(VIS_DIR, f'{name}-plot.png'))


def visualize():
    prepare_env()

    datasets = np.array(BINARY_CLASS_SETS)[:, 0]
    names = ['adaboost', 'catboost', 'lgb',
             'randomforest', 'xgboost', 'bagging']

    for name in names:
        df = pd.read_csv(result_path(
            name, out_dir='../EnsembleDiversityResults/experiments-10-08'))

        scatterplot(df, name + '__node-diversity',
                    x='node_diversity', datasets=datasets)
        scatterplot(df, name + '__used-attributes-ratio',
                    x='used_attributes_ratio', datasets=datasets)
        scatterplot(df, name + '__corr',
                    x='corr', datasets=datasets)
        scatterplot(df, name + '__q',
                    x='q', datasets=datasets)
        scatterplot(df, name + '__entropy',
                    x='entropy', datasets=datasets)
        scatterplot(df, name + '__kw',
                    x='kw', datasets=datasets)
        scatterplot(df, name + '__coverage-minmax',
                    x='coverage_minmax', datasets=datasets)
        scatterplot(df, name + '__coverage-std',
                    x='coverage_std', datasets=datasets)


if __name__ == '__main__':
    visualize()
