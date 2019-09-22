import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from typing import List
from measures import bin_corr, bin_q, bin_entropy, bin_df, bin_kw
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

    def avg_attributes_used(self) -> float:
        acc = []

        for tree in self.trees:
            acc.append(len(tree.attributes_used()))

        return np.mean(acc)

    def used_attributes_ratio(self) -> float:
        if self.dataset is None:
            return -1

        attributes = set()

        for tree in self.trees:
            attributes.update(tree.attributes_used())

        return len(attributes) / self.dataset.num_features()

    def avg_node_count(self) -> float:
        node_counts = np.array([tree.num_nodes() for tree in self.trees])
        return node_counts.mean()

    def node_diversity(self) -> float:
        node_counts = np.array([tree.num_nodes() for tree in self.trees])
        return node_counts.std()

    def coverage_leaves_std(self) -> float:
        acc = []

        for coverage in self.get_coverage():
            leaves_dict = coverage.get_leaves_dict()
            sizes = np.array([len(items) for items in leaves_dict.values()])

            sum_sizes = np.sum(sizes)
            if sum_sizes != 0:
                acc.append((sizes / np.sum(sizes)).std())

        if len(acc) < 1:
            return 0

        return np.std(acc)

    def coverage_leaves_minmax(self) -> float:
        acc = []

        for coverage in self.get_coverage():
            if coverage.size() == 0:
                continue

            leaves_dict = coverage.get_leaves_dict()
            sizes = np.array([len(items) for items in leaves_dict.values()])
            sizes = sizes / np.sum(sizes)
            acc.append(sizes.max() - sizes.min())

        if len(acc) < 1:
            return 0

        return np.std(acc)

    def coverage_leaves_avg(self) -> float:
        acc = []

        for coverage in self.get_coverage():
            if coverage.size() == 0:
                continue

            leaves_dict = coverage.get_leaves_dict()
            sizes = np.array([len(items) for items in leaves_dict.values()])
            sizes = sizes / np.sum(sizes)
            acc.append(sizes.max() - sizes.min())

        if len(acc) < 1:
            return 0

        return np.mean(acc)

    def q(self, val_set: Dataset) -> float:
        results = []
        tree_preds = [tree.predict(val_set.X, labeled_result=True)
                      for tree in self.trees]

        for i in range(0, len(tree_preds)):
            for j in range(0, len(tree_preds)):
                if i == j:
                    continue

                results.append(bin_q(val_set.y, tree_preds[i], tree_preds[j]))

        if len(results) < 1:
            return 0

        return np.average(results)

    def df(self, val_set: Dataset) -> float:
        results = []
        tree_preds = [tree.predict(val_set.X, labeled_result=True)
                      for tree in self.trees]

        for i in range(0, len(tree_preds)):
            for j in range(0, len(tree_preds)):
                if i == j:
                    continue

                results.append(bin_df(val_set.y, tree_preds[i], tree_preds[j]))

        if len(results) < 1:
            return 0

        return np.average(results)

    def corr(self, val_set: Dataset) -> float:
        results = []
        tree_preds = [tree.predict(val_set.X, labeled_result=True)
                      for tree in self.trees]

        for i in range(0, len(tree_preds)):
            for j in range(0, len(tree_preds)):
                if i == j:
                    continue

                results.append(
                    bin_corr(val_set.y, tree_preds[i], tree_preds[j]))

        if len(results) < 1:
            return 0

        return np.average(results)

    def entropy(self, val_set: Dataset) -> float:
        tree_preds = [tree.predict(val_set.X, labeled_result=True)
                      for tree in self.trees]
        return bin_entropy(val_set.y, tree_preds)

    def kw(self, val_set: Dataset) -> float:
        tree_preds = [tree.predict(val_set.X, labeled_result=True)
                      for tree in self.trees]
        return bin_kw(val_set.y, tree_preds)

    def clf_predict(self, dataset: Dataset) -> np.ndarray:
        if self.clf is None:
            raise ValueError('"clf" should be fitted')

        return self.clf.predict(dataset.X)

    @abstractmethod
    def fit(self, dataset: Dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Dataset) -> np.ndarray:
        raise NotImplementedError
