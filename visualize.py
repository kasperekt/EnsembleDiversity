import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import OUT_DIR, VIS_DIR, prepare_env


def result_path(name: str):
    filename = f'{name.lower()}experiment-ensemble.csv'
    return os.path.join(OUT_DIR, filename)


def scatterplot(df, name, datasets=['iris', 'cancer']):
    fig, axes = plt.subplots(ncols=len(datasets), figsize=(14, 5))

    for ax, dataset_name in zip(axes, datasets):
        data_df = df[df['dataset_name'] == dataset_name]
        x_name, y_name = 'node_diversity', 'accuracy'

        values = data_df[[x_name, y_name]].values
        xs, ys = values[:, 0], values[:, 1]

        ax.set_title(dataset_name)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.scatter(xs, ys)

    fig.suptitle(name)
    fig.savefig(os.path.join(VIS_DIR, f'{name}-plot.png'))


def visualize():
    prepare_env()

    datasets = ['iris', 'cancer']
    names = ['adaboost', 'catboost', 'lgb', 'randomforest']

    for name in names:
        df = pd.read_csv(result_path(name))
        scatterplot(df, name, datasets=datasets)


if __name__ == '__main__':
    visualize()
