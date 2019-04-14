import os
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from parser.lightgbm import parse_lightgbm
from parser.sklearn import parse_sklearn
from structure.DatasetStructure import DatasetStructure
from structure.TreeStructure import TreeStructure
from typing import List


def get_lgb_trees(dataset: DatasetStructure):
    clf = lgb.LGBMClassifier(n_estimators=5, objective='multiclass')
    clf.fit(dataset.X, dataset.y)

    return parse_lightgbm(clf, dataset)


def get_sklearn_trees(dataset: DatasetStructure):
    tree = DecisionTreeClassifier(max_depth=2)
    clf = AdaBoostClassifier(base_estimator=tree, n_estimators=5)
    clf.fit(dataset.X, dataset.y)

    return parse_sklearn(clf, dataset)


def main():
    iris = load_iris()
    iris_data = DatasetStructure(iris.data, iris.target, iris.feature_names, iris.target_names)

    tree_structures_sklearn = get_sklearn_trees(iris_data)
    tree_structures_lgb = get_lgb_trees(iris_data)

    structures: List[TreeStructure] = tree_structures_sklearn + tree_structures_lgb

    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join('plots', f'tree_{tree_structure.clf_type}_{i + 1}.png')
        tree_structure.draw(tree_path)


if __name__ == '__main__':
    main()
