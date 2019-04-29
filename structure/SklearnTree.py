import numpy as np

from .Tree import Tree
from .Dataset import Dataset


class SklearnTree(Tree):
    def __init__(self, dataset: Dataset, clf_type="Sklearn"):
        super().__init__(dataset, clf_type=clf_type)

    @staticmethod
    def parse(dataset: Dataset):
        tree = SklearnTree(dataset)
        return tree

    def add_leaf(self, idx, target=-1, fraction=0.0):
        self.tree.add_node(self.leaf_name(idx), target=target, fraction=fraction, is_leaf=True)

    def leaf_label(self, node_data: dict) -> str:
        target = node_data['target']
        target_name = self.dataset.target_names[target] if target != -1 else 'n/d'
        return f"{target_name}\n{node_data['fraction']}"

    # TODO: Move "recurse" to method
    def predict(self, data: np.ndarray) -> np.ndarray:
        root_idx = self.node_name(0)

        def recurse(X, node_idx):
            node = self.tree.nodes[node_idx]
            threshold = node['threshold']
            feature = node['feature']

            left_child_idx, right_child_idx = list(self.tree.successors(node_idx))

            if X[feature] <= threshold:
                child = self.tree.nodes[left_child_idx]

                if child['is_leaf']:
                    return child['target']

                return recurse(X, left_child_idx)
            else:
                child = self.tree.nodes[right_child_idx]

                if child['is_leaf']:
                    return child['target']

                return recurse(X, right_child_idx)

        return np.array([recurse(X, root_idx) for X in data], dtype=np.int)
