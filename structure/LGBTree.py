import numpy as np

from .Tree import Tree
from .Dataset import Dataset
from .utils import is_leaf, is_split


def node_index_of(child: dict):
    if is_leaf(child):
        return child['leaf_index']
    elif is_split(child):
        return child['split_index']


class LGBTree(Tree):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset, clf_type="LightGBM")

    @staticmethod
    def parse(tree: dict, dataset: Dataset):
        return_tree = LGBTree(dataset)

        def traverse(structure):
            if is_leaf(structure):
                return_tree.add_leaf(structure['leaf_index'],
                                     structure['leaf_value'])
            elif is_split(structure):
                split_index = structure['split_index']
                decision_type = structure['decision_type']
                split_feature = structure['split_feature']
                threshold = structure['threshold']

                return_tree.add_split(
                    split_index, decision_type, split_feature, threshold)

                left_child = structure['left_child']
                right_child = structure['right_child']

                traverse(left_child)
                traverse(right_child)

                return_tree.add_edge(node_index_of(structure),
                                     node_index_of(left_child),
                                     is_child_leaf=is_leaf(left_child))
                return_tree.add_edge(node_index_of(structure),
                                     node_index_of(right_child),
                                     is_child_leaf=is_leaf(right_child))

        traverse(tree['tree_structure'])

        return return_tree

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

            left_child_idx, right_child_idx = list(
                self.tree.successors(node_idx))

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
