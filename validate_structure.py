import sys
import numpy as np

from typing import Tuple
from structure import Dataset, AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble, CatboostEnsemble
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

Data = Tuple[Dataset, Dataset]


def validate(ValidatorType, name, data: Data, param_grid: ParameterGrid, verbose=False):
    train_data, val_data = data
    results = []

    for params in param_grid:
        ensemble = ValidatorType(params)
        ensemble.fit(train_data)

        preds = ensemble.predict(val_data.X)
        clf_preds = ensemble.clf_predict(val_data.X)

        accuracy = accuracy_score(preds, clf_preds)
        results.append(accuracy)

        if verbose and accuracy < 1.0:
            print(
                f'[{accuracy}] {name} for dataset "{train_data.name}" is not valid for params: {params}')

    general_acc = sum(results)/len(results)
    print(f'{name}, dataset={train_data.name}, acc={general_acc}')


def validate_ada(*args, **kwargs):
    return validate(AdaboostEnsemble, 'AdaBoost', *args, **kwargs)


def validate_rf(*args, **kwargs):
    return validate(RandomForestEnsemble, 'RandomForest', *args, **kwargs)


def validate_lgb(*args, **kwargs):
    return validate(LGBEnsemble, 'LGB', *args, **kwargs)


def validate_cb(*args, **kwargs):
    return validate(CatboostEnsemble, 'Catboost', *args, **kwargs)


def validate_structure(used_validators={'ada', 'rf'}, verbose=False):
    iris = Dataset.create_iris().split(test_size=0.5)
    cancer = Dataset.create_cancer().split(test_size=0.5)
    datasets = [iris, cancer]

    param_grid = ParameterGrid({
        'max_depth': range(2, 5),
        'n_estimators': range(1, 5)
    })

    validators = {
        'ada': validate_ada,
        'rf': validate_rf,
        'lgb': validate_lgb,
        'cb': validate_cb
    }

    for validator_key in used_validators:
        if validator_key not in validators:
            print(f'{validator_key} is not available', file=sys.stderr)
            return

        validator = validators[validator_key]

        for dataset in datasets:
            validator(dataset, param_grid, verbose)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-c', '--check', action='append', required=True)
    args = parser.parse_args()

    args_dict = {
        'verbose': args.verbose,
        'used_validators': args.check
    }

    validate_structure(**args_dict)
