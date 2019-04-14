import os
import lightgbm as lgb
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from parser.lightgbm import parse_lightgbm
from parser.sklearn import parse_sklearn
from structure.DatasetStructure import DatasetStructure
from structure.TreeStructure import TreeStructure
from typing import List


def get_lgb_trees(dataset: DatasetStructure):
    clf = lgb.LGBMClassifier(n_estimators=100, objective='multiclass')
    clf.fit(dataset.X, dataset.y)

    return parse_lightgbm(clf, dataset)


def get_adaboost_trees(dataset: DatasetStructure):
    tree = DecisionTreeClassifier(max_depth=5)
    clf = AdaBoostClassifier(base_estimator=tree, n_estimators=100)
    clf.fit(dataset.X, dataset.y)

    return parse_sklearn(clf, dataset, "AdaBoost")


def get_rf_trees(dataset: DatasetStructure):
    clf = RandomForestClassifier(max_depth=5, n_estimators=100)
    clf.fit(dataset.X, dataset.y)

    return parse_sklearn(clf, dataset, "RandomForest")


def draw_trees(structures: List[TreeStructure]):
    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join('plots', f'tree_{tree_structure.clf_type.lower()}_{i + 1}.png')
        tree_structure.draw(tree_path)


def main():
    iris = load_iris()
    iris_data = DatasetStructure(iris.data, iris.target, iris.feature_names, iris.target_names)

    tree_structures_lgb = get_lgb_trees(iris_data)
    tree_structures_adaboost = get_adaboost_trees(iris_data)
    tree_structures_rf = get_rf_trees(iris_data)

    draw_trees(tree_structures_lgb)
    draw_trees(tree_structures_adaboost)
    draw_trees(tree_structures_rf)

    tree_sizes_lgb = np.array([n.num_nodes() for n in tree_structures_lgb])
    tree_sizes_adaboost = np.array([n.num_nodes() for n in tree_structures_adaboost])
    tree_sizes_rf = np.array([n.num_nodes() for n in tree_structures_rf])

    print(f'Light GBM var = {np.std(tree_sizes_lgb)}')
    print(f'AdaBoost var = {np.std(tree_sizes_adaboost)}')
    print(f'Random forest var = {np.std(tree_sizes_rf)}')


if __name__ == '__main__':
    main()
