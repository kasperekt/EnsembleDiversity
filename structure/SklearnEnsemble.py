import numpy as np

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from parser.sklearn import parse_tree
from typing import Union
from .Dataset import Dataset


class SklearnEnsemble:
    def __init__(self, clf: Union[AdaBoostClassifier, BaggingClassifier], name='Sklearn'):
        self.trees = []
        self.clf = clf
        self.name = name

    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)

        self.trees = []

        for tree in self.clf.estimators_:
            tree_structure = parse_tree(tree, dataset, self.name)
            self.trees.append(tree_structure)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        preds = np.array([tree.predict(data) for tree in self.trees])

        return preds
