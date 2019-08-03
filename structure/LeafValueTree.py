import numpy as np

from scipy.special import expit
from .Tree import Tree


class LeafValueTree(Tree):
    def __init__(self, dataset, clf_type='n/d'):
        super().__init__(dataset, clf_type=clf_type)

    def leaf_label(self, node_data):
        return str(node_data['value'])

    def leaf_node_result(self, leaf_node):
        return leaf_node['value']

    def traverse(self, X, node_idx):
        if len(self.tree.nodes) == 0:
            return 0

        if len(self.tree.nodes) == 1:
            leaf_idx = self.leaf_name(0)
            leaf_node = self.tree.nodes[leaf_idx]

            return self.leaf_node_result(leaf_node)

        node = self.tree.nodes[node_idx]
        threshold = node['threshold']
        feature = node['feature']
        decision_type = node['decision_type']

        left_child_idx, right_child_idx = list(
            self.tree.successors(node_idx))

        if self.satisfies_cond(decision_type, X[feature], threshold):
            child = self.tree.nodes[left_child_idx]

            if child['is_leaf']:
                return self.leaf_node_result(child)

            return self.traverse(X, left_child_idx)
        else:
            child = self.tree.nodes[right_child_idx]

            if child['is_leaf']:
                return self.leaf_node_result(child)

            return self.traverse(X, right_child_idx)

    def predict(self, data: np.ndarray, labeled_result=False) -> np.ndarray:
        root_idx = self.node_name(0)

        preds = np.array([self.traverse(X, root_idx)
                          for X in data], dtype=np.float)

        if labeled_result:
            proba = np.array([[1 - v, v] for v in expit(preds)])
            return np.argmax(proba, axis=1)

        return preds

    def __repr__(self):
        return f'LeafValueTree<nodes_len={len(self.tree.nodes)}>'
