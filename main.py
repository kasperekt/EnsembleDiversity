import os

from sklearn.datasets import load_iris
from typing import List

from structure.Dataset import Dataset
from structure.Tree import Tree
from extract import get_rf_trees, get_lgb_trees, get_adaboost_trees


def draw_trees(structures: List[Tree]):
    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join('plots', f'tree_{tree_structure.clf_type.lower()}_{i + 1}.png')
        tree_structure.draw(tree_path)


def draw():
    iris = load_iris()
    iris_data = Dataset(iris.data, iris.target, iris.feature_names, iris.target_names)

    n = 10
    tree_structures_lgb = get_lgb_trees(iris_data, n_estimators=n, max_depth=2)
    tree_structures_adaboost = get_adaboost_trees(iris_data, n_estimators=n, max_depth=2)
    tree_structures_rf = get_rf_trees(iris_data, n_estimators=n, max_depth=2)

    draw_trees(tree_structures_lgb)
    draw_trees(tree_structures_adaboost)
    draw_trees(tree_structures_rf)


if __name__ == '__main__':
    draw()
