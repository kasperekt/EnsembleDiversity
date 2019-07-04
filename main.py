import os

from experiments import LGBExperiment, AdaboostExperiment, RandomForestExperiment, XGBoostExperiment, CatboostExperiment, BaggingExperiment
from data import Dataset, load_all_datasets
from structure import AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble
from config import OUT_DIR, prepare_env
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def run_experiment():
    train_datasets, val_datasets = load_all_datasets(test_size=0.5)

    experiments = [
        AdaboostExperiment(),
        RandomForestExperiment(),
        BaggingExperiment(),
        LGBExperiment(),
        CatboostExperiment(),
        XGBoostExperiment(),
    ]

    for exp in experiments:
        exp.run(train_datasets, val_datasets)
        exp.to_csv(os.path.join(OUT_DIR, f'{exp.name.lower()}-ensemble.csv'))


if __name__ == '__main__':
    prepare_env()
    run_experiment()
