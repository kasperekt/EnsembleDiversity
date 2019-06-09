import numpy as np

from .utils import node_index_of, is_leaf, is_split
from structure import LeafValueTree
from data import Dataset


class LGBTree(LeafValueTree):
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
