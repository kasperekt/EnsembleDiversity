import os
import pandas as pd

NAMES = ['bagging', 'adaboost', 'randomforest',
         'lgb', 'catboost', 'xgboost']


def csv_path(base_dir, name):
    return os.path.join(
        base_dir, f'{name}experiment-ensemble.csv')


def read_csv(base_dir: str, classifier_name: str):
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(
            f'{base_dir} is not a directory (or it does not exist.)')

    if classifier_name not in NAMES:
        raise AttributeError(f'Wrong classifier name {classifier_name}')

    df_path = csv_path(base_dir, classifier_name)

    if not os.path.isfile(df_path):
        print(f'{classifier_name} experiment does not exist')

    return pd.read_csv(df_path, index_col=0)
