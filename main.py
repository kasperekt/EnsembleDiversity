import os
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from parser.lightgbm import parse_lightgbm
from parser.sklearn import parse_sklearn
from structure.TreeStructure import TreeStructure
from typing import List


def get_lgb_trees(X, y, **kwargs):
    clf = lgb.LGBMClassifier(n_estimators=5, objective='multiclass')
    clf.fit(X, y)

    return parse_lightgbm(clf, **kwargs)


def get_sklearn_trees(X, y, **kwargs):
    tree = DecisionTreeClassifier(max_depth=3)
    clf = AdaBoostClassifier(base_estimator=tree, n_estimators=5)
    clf.fit(X, y)

    return parse_sklearn(clf, **kwargs)


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    additional_data = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }

    tree_structures_sklearn = get_sklearn_trees(X, y, **additional_data)
    tree_structures_lgb = get_lgb_trees(X, y, **additional_data)

    structures: List[TreeStructure] = tree_structures_sklearn + tree_structures_lgb

    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join('plots', f'tree_{tree_structure.clf_type}_{i + 1}.png')
        tree_structure.draw(tree_path)


if __name__ == '__main__':
    main()
