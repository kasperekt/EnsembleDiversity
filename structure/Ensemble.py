import numpy as np

from abc import ABCMeta, abstractmethod
from typing import List
from data import Dataset
from .Tree import Tree


class Ensemble(metaclass=ABCMeta):
    def __init__(self, params: dict, name="Ensemble"):
        self.trees = []
        self.params = params
        self.name = name
        self.clf = None

    def __iter__(self):
        for tree in self.trees:
            yield tree

    def node_diversity(self) -> float:
        node_counts = np.array([tree.num_nodes() for tree in self.trees])
        return node_counts.std()

    def clf_predict(self, dataset: Dataset) -> np.ndarray:
        if self.clf is None:
            raise ValueError('"clf" should be fitted')

        return self.clf.predict(dataset.X)

    @abstractmethod
    def fit(self, dataset: Dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
