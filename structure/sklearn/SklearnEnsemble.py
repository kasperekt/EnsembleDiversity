import numpy as np

from abc import ABCMeta
from predict import majority_voting
from data import Dataset, DatasetEncoder
from structure import Ensemble
from . import SklearnTree


class SklearnEnsemble(Ensemble, metaclass=ABCMeta):
    def __init__(self, params: dict, name='Sklearn'):
        super().__init__(params, name)
        self.clf = None
        self.dataset_encoder = None

    def fit(self, dataset: Dataset):
        if self.dataset_encoder is None:
            self.dataset_encoder = DatasetEncoder.create_one_hot(dataset)

        encoded_dataset = self.dataset_encoder.transform(dataset)

        self.clf.fit(encoded_dataset.X, encoded_dataset.y)
        self.trees = [SklearnTree.parse(tree, encoded_dataset)
                      for tree in self.clf.estimators_]

    def predict(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        if self.dataset_encoder is None:
            raise ValueError('No dataset encoder available')

        encoded_dataset = self.dataset_encoder.transform(dataset)

        predictions = np.array([tree.predict(encoded_dataset.X)
                                for tree in self.trees])
        predictions = np.rollaxis(predictions, axis=1)

        return majority_voting(predictions)
