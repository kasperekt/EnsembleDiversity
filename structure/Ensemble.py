import numpy as np

from abc import ABCMeta, abstractmethod
from typing import List
from .Tree import Tree
from .Dataset import Dataset


class Ensemble(metaclass=ABCMeta):
    def __init__(self, params: dict, name="Ensemble"):
        self.trees = []
        self.params = params
        self.name = name
        self.clf = None

    def __iter__(self):
        for tree in self.trees:
            yield tree

    def fit(self, dataset: Dataset):
        raise NotImplementedError

    def node_diversity(self) -> float:
        node_counts = np.array([tree.num_nodes() for tree in self.trees])
        return node_counts.std()

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
