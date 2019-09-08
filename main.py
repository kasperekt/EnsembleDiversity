import os

import itertools
import experiments.diversity as div
import experiments.rank as rank

from data import Dataset, load_all_datasets
from config import OUT_DIR, prepare_env
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def run_experiment(variant: str, cv: bool, experiment: str, repetitions: int):
    if experiment == 'diversity':
        experiments = div.load_all_experiments(variant)
    elif experiment == 'rank':
        experiments = rank.load_all_experiments(variant)

    datasets = load_all_datasets()

    if cv > 1:
        datasets = list(itertools.chain.from_iterable(
            [dataset.n_splits(cv) for dataset in datasets]))
    else:
        datasets = [dataset.split(0.2) for dataset in datasets]

    for exp in experiments:
        exp.run(datasets, repetitions=repetitions)
        exp.to_csv(os.path.join(OUT_DIR, f'{exp.name.lower()}-ensemble.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='shared',
                        choices=['individual', 'shared'])
    parser.add_argument('--experiment', default='diversity',
                        choices=['rank', 'diversity'])
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--reps', type=int, default=1)

    args = parser.parse_args()

    prepare_env()
    run_experiment(args.variant, args.cv, args.experiment, args.reps)
