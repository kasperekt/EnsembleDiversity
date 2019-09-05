import os

import experiments.diversity as div
import experiments.rank as rank

from data import Dataset, load_all_datasets
from config import OUT_DIR, prepare_env
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def run_experiment(variant: str, cv: bool, experiment: str):
    if experiment == 'diversity':
        experiments = div.load_all_experiments(variant, cv)
    elif experiment == 'rank':
        experiments = rank.load_all_experiments(variant, cv)

    for exp in experiments:
        exp.run(load_all_datasets())
        exp.to_csv(os.path.join(OUT_DIR, f'{exp.name.lower()}-ensemble.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='shared',
                        choices=['individual', 'shared'])
    parser.add_argument('--experiment', default='diversity',
                        choices=['rank', 'diversity'])
    parser.add_argument('--cv', action='store_true')

    args = parser.parse_args()

    prepare_env()
    run_experiment(args.variant, args.cv, args.experiment)
