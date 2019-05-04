import numpy as np

from typing import Union
from predict import majority_voting
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from .Dataset import Dataset
from .SklearnTree import SklearnTree


class SklearnEnsemble:
    def __init__(self, clf: Union[AdaBoostClassifier, BaggingClassifier], name='Sklearn'):
        self.trees = []
        self.clf = clf
        self.name = name

    def __iter__(self):
        for tree in self.trees:
            yield tree

    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)
        self.trees = [SklearnTree.parse(tree, dataset) for tree in self.clf.estimators_]

    def predict(self, data: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        # TODO: Doesn't work - very different predictions
        predictions = np.array([tree.predict(data) for tree in self.trees])
        predictions = np.rollaxis(predictions, axis=1)

        return majority_voting(predictions)
