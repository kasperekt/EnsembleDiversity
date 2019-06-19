import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import OUT_DIR, VIS_DIR, prepare_env


def result_path(name: str, out_dir=OUT_DIR):
    filename = f'{name.lower()}experiment-ensemble.csv'
    return os.path.join(out_dir, filename)


def scatterplot(df, name, x, y='accuracy', datasets=['iris', 'cancer']):
    fig, axes = plt.subplots(ncols=len(datasets), figsize=(14, 5))

    for ax, dataset_name in zip(axes, datasets):
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

    datasets = ['iris', 'cancer']
    names = ['adaboost', 'catboost', 'lgb',
             'randomforest', 'xgboost', 'bagging']

    for name in names:
        df = pd.read_csv(result_path(
            name, out_dir='../EnsembleDiversityResults/experiments-16-06'))

        scatterplot(df, name + '__node-diversity',
                    x='node_diversity', datasets=datasets)
        scatterplot(df, name + '__attr-diversity',
                    x='attr_diversity', datasets=datasets)
        scatterplot(df, name + '__feature-diversity',
                    x='feature_diversity', datasets=datasets)


if __name__ == '__main__':
    visualize()
