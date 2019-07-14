import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from typing import List
from data import Dataset, DatasetEncoder
from .Tree import Tree
from .TreeCoverage import TreeCoverage


class Ensemble(metaclass=ABCMeta):
    def __init__(self, params: dict, dataset: Dataset = None, name='Ensemble'):
        self.trees = []
        self.params = params
        self.name = name
        self.clf = None
        self.dataset = None
        self.dataset_encoder = None

        if dataset:
            self.set_dataset(dataset)

    def __iter__(self):
        for tree in self.trees:
            yield tree

    def set_dataset(self, dataset: Dataset):
        self.dataset_encoder = self.create_encoder(dataset)
        self.dataset = self.encode_dataset(dataset)

    def create_encoder(self, dataset: Dataset):
        if self.dataset_encoder is None:
            de = DatasetEncoder.create_one_hot(dataset)
            return de

        return self.dataset_encoder

    def get_coverage(self) -> List[TreeCoverage]:
        if self.dataset is None:
            raise TypeError('No dataset avaialable')

        return [TreeCoverage.parse(tree, self.dataset) for tree in self.trees]

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
