import os
import argparse
import pandas as pd

from thesis.io import read_csv, csv_path

ALGORITHMS = ['bagging', 'adaboost',
              'randomforest', 'lgb', 'catboost', 'xgboost']


def main(args):
    data_dir = args.dir

    dfs = [read_csv(data_dir, name) for name in ALGORITHMS]
    all_df = pd.concat(dfs)

    all_df_path = csv_path(data_dir, 'all')
    print(f'Concatenating DFs. Writing to {all_df_path}')
    all_df.to_csv(all_df_path)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    main(args)
