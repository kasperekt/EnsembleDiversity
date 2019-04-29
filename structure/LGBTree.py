import numpy as np

from .Tree import Tree
from .Dataset import Dataset


class LGBTree(Tree):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset, clf_type="LightGBM")

    def add_leaf(self, idx, value=0.0):
        self.tree.add_node(self.leaf_name(idx), value=value, is_leaf=True)

    def leaf_label(self, node_data: dict) -> str:
        return f"{node_data['value']}"

    def predict(self, data: np.ndarray) -> np.ndarray:
        root_idx = self.node_name(0)

        def recurse(X, node_idx):
            node = self.tree.nodes[node_idx]
            threshold = node['threshold']
            feature = node['feature']

            left_child_idx, right_child_idx = list(self.tree.successors(node_idx))

            # TODO: Use decision_type info for condition
            if X[feature] <= threshold:
                child = self.tree.nodes[left_child_idx]

                if child['is_leaf']:
                    return child['value']

                return recurse(X, left_child_idx)
            else:
                child = self.tree.nodes[right_child_idx]

                if child['is_leaf']:
                    return child['value']

                return recurse(X, right_child_idx)

        return np.array([recurse(X, root_idx) for X in data], dtype=np.float)
