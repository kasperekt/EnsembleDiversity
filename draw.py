import os

from typing import List
from structure import Dataset, Tree, AdaboostEnsemble, LGBEnsemble, RandomForestEnsemble, CatboostEnsemble
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

    Types = [LGBEnsemble, AdaboostEnsemble,
             RandomForestEnsemble, CatboostEnsemble]

    for Type in Types:
        ensemble = Type(general_params)
        ensemble.fit(iris_data)
        draw_trees(ensemble.trees)


if __name__ == '__main__':
    draw()
