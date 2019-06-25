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
        encoded_dataset = dataset.oh_encoded()

        self.clf.fit(encoded_dataset.X, encoded_dataset.y)
        self.trees = [SklearnTree.parse(tree, encoded_dataset)
                      for tree in self.clf.estimators_]

    def predict(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        predictions = np.array([tree.predict(dataset.X)
                                for tree in self.trees])
        predictions = np.rollaxis(predictions, axis=1)

        return majority_voting(predictions)
