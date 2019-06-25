import sys
import numpy as np

from typing import Tuple
from data import Dataset
from structure import AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble, CatboostEnsemble, XGBoostEnsemble, BaggingEnsemble
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

Data = Tuple[Dataset, Dataset]


def validate(ValidatorType, name, data: Data, param_grid: ParameterGrid, verbose=False):
    train_data, val_data = data

    compat_results = []
    pred_results = []
    clf_pred_results = []

    for params in param_grid:
        ensemble = ValidatorType(params)
        ensemble.fit(train_data)

        preds = ensemble.predict(val_data)
        clf_preds = ensemble.clf_predict(val_data)

        # Compatibility
        compat_accuracy = accuracy_score(preds, clf_preds)
        compat_results.append(compat_accuracy)

        # Structure prediction accuracy
        pred_accuracy = accuracy_score(val_data.y, preds)
        pred_results.append(pred_accuracy)

        # Classifier prediction accuracy
        clf_pred_accuracy = accuracy_score(val_data.y, clf_preds)
        clf_pred_results.append(clf_pred_accuracy)

    compat_general_acc = sum(compat_results)/len(compat_results)
    pred_general_acc = sum(pred_results)/len(pred_results)
    clf_pred_general_acc = sum(clf_pred_results)/len(clf_pred_results)

    if verbose:
        print(f'{name}, dataset={train_data.name}')
        print(f'compatibility acc={compat_general_acc}')
        print(f'prediction acc={pred_general_acc}')
        print(f'clf prediction acc={clf_pred_general_acc}')
        print()
    else:
        print(f'{name}, dataset={train_data.name}, acc={compat_general_acc}')


def validate_ada(*args, **kwargs):
    return validate(AdaboostEnsemble, 'AdaBoost', *args, **kwargs)


def validate_rf(*args, **kwargs):
    return validate(RandomForestEnsemble, 'RandomForest', *args, **kwargs)


def validate_lgb(*args, **kwargs):
    return validate(LGBEnsemble, 'LGB', *args, **kwargs)


def validate_cb(*args, **kwargs):
    return validate(CatboostEnsemble, 'Catboost', *args, **kwargs)


def validate_xgb(*args, **kwargs):
    return validate(XGBoostEnsemble, 'XGBoost', *args, **kwargs)


def validate_bag(*args, **kwargs):
    return validate(BaggingEnsemble, 'Bagging', *args, **kwargs)


VALIDATORS = {
    'ada': validate_ada,
    'rf': validate_rf,
    'lgb': validate_lgb,
    'cb': validate_cb,
    'xgb': validate_xgb,
    'bag': validate_bag
}


def validate_structure(used_validators={'ada', 'rf'}, verbose=False):
    iris = Dataset.create_iris().split(test_size=0.8)
    cancer = Dataset.create_cancer().split(test_size=0.8)
    aids = Dataset.from_openml('aids').split(test_size=0.8)
    datasets = [iris, cancer, aids]

    param_grid = ParameterGrid({
        'max_depth': range(2, 10),
        'n_estimators': range(1, 100, 10)
    })

    for validator_key in used_validators:
        if validator_key not in VALIDATORS:
            print(f'{validator_key} is not available', file=sys.stderr)
            return

        validator = VALIDATORS[validator_key]

        for dataset in datasets:
            validator(dataset, param_grid, verbose)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-c', '--check', action='append')
    parser.add_argument('-A', '--all', action='store_true')
    args = parser.parse_args()

    used_validators = list(VALIDATORS.keys()) if args.all else args.check

    args_dict = {
        'verbose': args.verbose,
        'used_validators': used_validators
    }

    validate_structure(**args_dict)
