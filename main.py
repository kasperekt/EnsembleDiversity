import os

from experiments import LGBExperiment, AdaboostExperiment, RandomForestExperiment, XGBoostExperiment, CatboostExperiment, BaggingExperiment
from data import Dataset, load_all_datasets
from structure import AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble
from config import OUT_DIR, prepare_env
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def run_experiment(variant: str):
    train_datasets, val_datasets = load_all_datasets(test_size=0.5)

    experiments = [
        AdaboostExperiment(variant),
        RandomForestExperiment(variant),
        BaggingExperiment(variant),
        LGBExperiment(variant),
        CatboostExperiment(variant),
        XGBoostExperiment(variant),
    ]

    for exp in experiments:
        exp.run(train_datasets, val_datasets)
        exp.to_csv(os.path.join(OUT_DIR, f'{exp.name.lower()}-ensemble.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='shared',
                        choices=['individual', 'shared'])

    args = parser.parse_args()

    prepare_env()
    run_experiment(args.variant)
