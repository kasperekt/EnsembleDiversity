import numpy as np

from sklearn.tree import DecisionTreeClassifier
from .Tree import Tree
from .Dataset import Dataset


class SklearnTree(Tree):
    def __init__(self, dataset: Dataset, clf_type="Sklearn"):
        super().__init__(dataset, clf_type=clf_type)

    @staticmethod
    def parse(decision_tree: DecisionTreeClassifier, dataset: Dataset):
        tree_structure = SklearnTree(dataset, 'Sklearn')

        children_left = decision_tree.tree_.children_left
        children_right = decision_tree.tree_.children_right
        threshold = decision_tree.tree_.threshold
        features = decision_tree.tree_.feature
        value = decision_tree.tree_.value

        def traverse(node: int):
            if threshold[node] != -2:
                tree_structure.add_split(node,
                                         decision_type='<=',
                                         feature=features[node],
                                         threshold=threshold[node])

                if children_left[node] != -1:
                    left_child = children_left[node]
                    traverse(left_child)
                    tree_structure.add_edge(node,
                                            left_child,
                                            is_child_leaf=threshold[left_child] == -2)

                if children_right[node] != -1:
                    right_child = children_right[node]
                    traverse(right_child)
                    tree_structure.add_edge(node,
                                            right_child,
                                            is_child_leaf=threshold[right_child] == -2)
            else:
                target = value[node]
                for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                    # TODO: last param in add_leaf should be count
                    tree_structure.add_leaf(node, target=i, fraction=v)

        traverse(0)

        return tree_structure

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
