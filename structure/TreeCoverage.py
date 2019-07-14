import sys
import numpy as np

from data import Dataset
from structure import Tree


class TreeCoverage(object):
    def __init__(self, dataset: Dataset, coverage_dict: dict = {}):
        self.dataset = dataset
        self.coverage_dict: dict = coverage_dict

    def __repr__(self):
        return str(self.coverage_dict)

    def add_entry(self, node_idx: str, values: np.ndarray):
        self.coverage_dict[node_idx] = values

    def get_node(self, node_idx) -> np.ndarray:
        return self.coverage_dict[node_idx]

    def get_leaves_dict(self) -> dict:
        leaves = {}

        for key, value in self.coverage_dict.items():
            if 'Leaf' in key:
                leaves[key] = value.copy()

        return leaves

    def leaves(self):
        for key, value in self.get_leaves_dict().items():
            yield key, value

    @staticmethod
    def parse(tree: Tree, dataset: Dataset):
        if len(tree.tree) < 2:
            raise NotImplementedError

        root_idx = tree.node_name(0)
        original_indices = np.array(list(range(0, len(dataset.X))))

        def traverse(node_idx, indices):
            node = tree.tree.nodes[node_idx]
            children = list(tree.tree.successors(node_idx))

            if node['is_leaf']:
                return {node_idx: indices}

            decision_type = node['decision_type']
            feature_idx = node['feature']
            threshold = node['threshold']

            X = dataset.X[indices]
            X_indices = tree.satisfies_cond(
                decision_type, X[:, feature_idx], threshold)

            l_indices = np.argwhere(X_indices == True).flatten()
            r_indices = np.argwhere(X_indices == False).flatten()

            l_dict = traverse(children[0], indices[l_indices])
            r_dict = traverse(children[1], indices[r_indices])

            return {
                node_idx: indices,
                **l_dict,
                **r_dict,
            }

        coverage_dict = traverse(root_idx, original_indices)

        return TreeCoverage(dataset, coverage_dict)
