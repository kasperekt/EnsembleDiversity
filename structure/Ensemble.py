import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from typing import List
from data import Dataset, DatasetEncoder
from .Tree import Tree


class Ensemble(metaclass=ABCMeta):
    def __init__(self, params: dict, name="Ensemble"):
        self.trees = []
        self.params = params
        self.name = name
        self.clf = None

        self.dataset_encoder = None

    def __iter__(self):
        for tree in self.trees:
            yield tree

    def create_encoder(self, dataset: Dataset):
        if self.dataset_encoder is None:
            de = DatasetEncoder.create_one_hot(dataset)
            self.dataset_encoder = de

    def encode_dataset(self, dataset: Dataset):
        if self.dataset_encoder is not None:
            return self.dataset_encoder.transform(dataset)

        warnings.warn(
            'ATTENTION! Cannot encode dataset. It won\'t have transformed features as one-hot vectors.')

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
