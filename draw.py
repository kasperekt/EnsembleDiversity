import os

from typing import List
from data import Dataset, BINARY_CLASS_SETS
from structure import Tree, AdaboostEnsemble, LGBEnsemble, RandomForestEnsemble, CatboostEnsemble, XGBoostEnsemble
from sklearn.datasets import load_iris


def draw_trees(structures: List[Tree], pretty: bool):
    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join(
            'plots', f'tree_{tree_structure.clf_type.lower()}_{i + 1}.png')
        tree_structure.draw(tree_path, pretty=pretty)


def draw(pretty: bool):
    set_id, set_version = BINARY_CLASS_SETS[8]  # Boston
    iris_data = Dataset.from_openml(set_id, set_version)

    Types = [LGBEnsemble, RandomForestEnsemble,
             CatboostEnsemble, XGBoostEnsemble]

    for Type in Types:
        params = {
            'n_estimators': 2,
            'max_depth': 3
        }

        ensemble = Type(params)
        ensemble.fit(iris_data)
        draw_trees(ensemble.trees, pretty)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p, --pretty', action='store_true', dest='pretty')

    args = parser.parse_args()

    draw(args.pretty)
