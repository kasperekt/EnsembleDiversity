import os

from typing import List
from structure import Dataset, Tree, AdaboostEnsemble, LGBEnsemble, RandomForestEnsemble
from sklearn.datasets import load_iris


def draw_trees(structures: List[Tree]):
    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join(
            'plots', f'tree_{tree_structure.clf_type.lower()}_{i + 1}.png')
        tree_structure.draw(tree_path)


def draw():
    iris_data = Dataset.from_sklearn("iris", load_iris())

    general_params = {
        'n_estimators': 10,
        'max_depth': 3
    }

    lgb_ensemble = LGBEnsemble(general_params)
    lgb_ensemble.fit(iris_data)

    adaboost_ensemble = AdaboostEnsemble(general_params)
    adaboost_ensemble.fit(iris_data)

    rf_ensemble = RandomForestEnsemble(general_params)
    rf_ensemble.fit(iris_data)

    draw_trees(lgb_ensemble.trees)
    draw_trees(adaboost_ensemble.trees)
    draw_trees(rf_ensemble.trees)


if __name__ == '__main__':
    draw()
