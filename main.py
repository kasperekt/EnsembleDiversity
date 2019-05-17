import os

from experiments import LGBExperiment, AdaboostExperiment, RandomForestExperiment
from structure import Tree, Dataset, AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble
from config import OUT_DIR
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def run_experiment():
    iris_train, iris_val = Dataset.from_sklearn("iris", load_iris()).split(0.5)
    cancer_train, cancer_val = Dataset.from_sklearn("cancer",
                                                    load_breast_cancer()).split(0.5)

    train_datasets = [iris_train, cancer_train]
    val_datasets = [iris_val, cancer_val]

    experiments = [LGBExperiment(), AdaboostExperiment(),
                   RandomForestExperiment()]

    for exp in experiments:
        exp.run(train_datasets, val_datasets)
        exp.to_csv(os.path.join(OUT_DIR, f'{exp.name.lower()}-ensemble.csv'))


if __name__ == '__main__':
    run_experiment()
