import numpy as np

from .Tree import Tree


class LeafValueTree(Tree):
    def __init__(self, dataset, clf_type='n/d'):
        super().__init__(dataset, clf_type=clf_type)

    def leaf_label(self, node_data):
        return str(node_data['value'])

    def predict(self, data: np.ndarray) -> np.ndarray:
        root_idx = self.node_name(0)

        def recurse(X, node_idx):
            if len(self.tree.nodes) == 0:
                # TODO: Check if this is valid
                return 0

            if len(self.tree.nodes) == 1:
                leaf_idx = self.leaf_name(0)
                leaf_node = self.tree.nodes[leaf_idx]
                return leaf_node['value']

            node = self.tree.nodes[node_idx]
            threshold = node['threshold']
            feature = node['feature']
            decision_type = node['decision_type']

            left_child_idx, right_child_idx = list(
                self.tree.successors(node_idx))

            if self.satisfies_cond(decision_type, X[feature], threshold):
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
