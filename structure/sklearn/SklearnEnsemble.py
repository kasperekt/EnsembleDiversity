import numpy as np

from abc import ABCMeta
from predict import majority_voting
from data import Dataset
from structure import Ensemble
from . import SklearnTree


class SklearnEnsemble(Ensemble, metaclass=ABCMeta):
    def __init__(self, params: dict, name='Sklearn'):
        super().__init__(params, name)
        self.clf = None

    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)
        self.trees = [SklearnTree.parse(tree, dataset)
                      for tree in self.clf.estimators_]

    def predict(self, data: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        # TODO: Doesn't work - very different predictions
        predictions = np.array([tree.predict(data) for tree in self.trees])
        predictions = np.rollaxis(predictions, axis=1)

        return majority_voting(predictions)
