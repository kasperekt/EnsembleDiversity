import numpy as np
import networkx as nx

from abc import ABCMeta, abstractmethod
from typing import List, Callable
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

    def node_diversity(self) -> float:
        node_counts = np.array([tree.num_nodes() for tree in self.trees])
        return node_counts.std()

    def pairwise_diversity(self,
                           distance_fn: Callable[[Tree, Tree], float],
                           result_fn: Callable[[List[float]], float]) -> float:
        n_trees = len(self.trees)
        results = []

        for i in range(n_trees):
            tree_i = self.trees[i]
            for j in range(i + 1, n_trees):
                tree_j = self.trees[j]

                difference = distance_fn(tree_i, tree_j)
                results.append(difference)

        return result_fn(results)

    def used_attrs_diversity(self) -> float:
        def distance(tree_i: Tree, tree_j: Tree) -> float:
            max_len = max(len(tree_i.used_attrs), len(tree_j.used_attrs))

            if max_len == 0:
                return 0

            intersection = len(
                tree_i.used_attrs.intersection(tree_j.used_attrs))

            return 1 - (intersection / max_len)

        def result(results):
            return np.array(results).std()

        return self.pairwise_diversity(distance, result)

    def used_feature_diversity(self) -> float:
        def distance(tree_i: Tree, tree_j: Tree) -> float:
            max_len = max(len(tree_i.used_features), len(tree_j.used_features))

            if max_len == 0:
                return 0

            intersection = len(
                tree_i.used_features.intersection(tree_j.used_features))

            return 1 - (intersection / max_len)

        def result(results):
            return np.array(results).std()

        return self.pairwise_diversity(distance, result)

    def clf_predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise ValueError('"clf" should be fitted')

        return self.clf.predict(X)

    @abstractmethod
    def fit(self, dataset: Dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return len(self.trees)
