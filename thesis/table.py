import os
import argparse
import pandas as pd
import numpy as np

from thesis.io import read_csv, NAMES
from data.sets import BINARY_CLASS_SETS

MEASURES = [
    # STRUCT.
    'node_diversity',
    'used_attributes_ratio',
    # BEHAV.
    'corr',
    'df',
    'entropy',
    'kw',
    'q',
    'coverage_minmax',
    'coverage_std'
]


def rank_table(base_dir):
    dfs = [(name, read_csv(base_dir, name)) for name in NAMES]
    dataset_names = [name for (name, _) in BINARY_CLASS_SETS]

    results = {}

    for estimator_name, df in dfs:
        for (n, dataset_name), group in df.groupby(['n_estimators', 'dataset_name']):
            acc = group['accuracy']
            best_acc = acc.max()

            estimator_id = f'{estimator_name}_{n}'

            if dataset_name not in results:
                results[dataset_name] = {}

            results[dataset_name][estimator_id] = best_acc

    results = pd.DataFrame(results).transpose()

    ranked_results = results.rank(ascending=1, axis=1, method='dense')

    return ranked_results


def df_from_groups(groups, with_avg=True):
    column_set = ['index']
    results = {}

    for (dataset_name, name), value in groups.items():
        if name not in column_set:
            column_set.append(name)

        if dataset_name not in results:
            results[dataset_name] = []

        results[dataset_name].append(value)

    results = np.array([[dataset_name, *values]
                        for dataset_name, values in results.items()])

    if with_avg:
        column_avgs = np.array(
            [['AVERAGE', *np.mean(results[:, 1:].astype(np.float), axis=0)]])
        results = np.concatenate((results, column_avgs), axis=0)

    return pd.DataFrame(results, columns=column_set)


def get_avg_table(base_dir, measure):
    all_df = read_csv(base_dir, 'all')
    groups = all_df.groupby(by=['dataset_name', 'name'])

    results = {}

    for group_name, group_df in groups:
        values = group_df[measure].values
        results[group_name] = np.mean(values)

    return df_from_groups(results)


def diversity_tables(base_dir):
    my_path = os.path.dirname(os.path.realpath(__file__))

    for measure in MEASURES:
        table = get_avg_table(base_dir, measure)
        table.to_html(os.path.join(my_path, f'tables/{measure}-table.html'))


def main(args):
    base_dir = args.dir

    if args.type == 'rank':
        rank_table(base_dir)
    elif args.type == 'diversity':
        diversity_tables(base_dir)
    else:
        raise NotImplementedError(f'{args.type} table is not implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--type', default='rank')
    args = parser.parse_args()
    main(args)
