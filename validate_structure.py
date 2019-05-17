import numpy as np

from typing import Tuple
from structure import Dataset, AdaboostEnsemble, RandomForestEnsemble
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

Data = Tuple[Dataset, Dataset]


def validate_ada(data: Data, param_grid: ParameterGrid, verbose=False):
    train_data, val_data = data
    results = []

    for params in param_grid:
        ensemble = AdaboostEnsemble(params)
        ensemble.fit(train_data)

        preds = ensemble.predict(val_data.X)
        clf_preds = ensemble.clf_predict(val_data.X)

        accuracy = accuracy_score(preds, clf_preds)
        results.append(accuracy)

        if verbose and accuracy < 1.0:
            print(
                f'[{accuracy}] AdaBoost for dataset "{train_data.name}" is not valid for params: {params}')

    general_acc = sum(results)/len(results)
    print(f'AdaBoost, dataset={train_data.name}, acc={general_acc}')


def validate_rf(data: Data, param_grid: ParameterGrid, verbose=False):
    train_data, val_data = data
    results = []

    for params in param_grid:
        ensemble = RandomForestEnsemble(params)
        ensemble.fit(train_data)

        preds = ensemble.predict(val_data.X)
        clf_preds = ensemble.clf_predict(val_data.X)

        accuracy = accuracy_score(preds, clf_preds)
        results.append(accuracy)

        if verbose and accuracy < 1.0:
            print(
                f'[{accuracy}] RandomForest for dataset "{train_data.name}" is not valid for params: {params}')

    general_acc = sum(results)/len(results)
    print(f'RandomForest, dataset={train_data.name}, acc={general_acc}')


def validate_structure(verbose=False):
    iris = Dataset.create_iris().split(test_size=0.5)
    cancer = Dataset.create_cancer().split(test_size=0.5)

    param_grid = ParameterGrid({
        'max_depth': range(2, 20),
        'n_estimators': range(1, 30)
    })

    validate_ada(iris, param_grid, verbose)
    validate_ada(cancer, param_grid, verbose)

    validate_rf(iris, param_grid, verbose)
    validate_rf(cancer, param_grid, verbose)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    args_dict = {
        'verbose': args.verbose
    }

    validate_structure(**args_dict)
