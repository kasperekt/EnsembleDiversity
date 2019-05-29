import numpy as np

from .Tree import Tree
from .Dataset import Dataset


def split_idx_generator():
    idx = 0
    while True:
        yield idx
        idx += 1


def divide_leaves(leaves, n_classes):
    new_shape = (n_classes, len(leaves) // n_classes)
    return np.rollaxis(leaves.reshape(new_shape), axis=1)


class CatboostTree(Tree):
    def __init__(self, dataset, clf_type='n/d'):
        super().__init__(dataset, clf_type=clf_type)

    def leaf_label(self, node_data):
        return node_data['value']

    @staticmethod
    def parse(tree: dict, dataset: Dataset):
        return_tree = CatboostTree(dataset)

        split_idx = split_idx_generator()

        splits = tree['splits']
        n_classes = dataset.num_classes()
        leaf_values = np.array(tree['leaf_values'])
        leaf_weights = np.array(tree['leaf_weights'])

        if n_classes > 2:
            leaf_values = divide_leaves(leaf_values, n_classes)

        leaf_indices = np.arange(0, len(leaf_values))

        def traverse(li, parent_idx=None, it=0):
            if len(li) == 1:
                leaf_index = li[0]

                return_tree.add_leaf(leaf_index,
                                     value=leaf_values[leaf_index],
                                     count=leaf_weights[leaf_index])

                if parent_idx is not None:
                    return_tree.add_edge(
                        parent_idx, leaf_index, is_child_leaf=True)

                return leaf_index

            split = splits[it]
            feature_idx = int(split['float_feature_index'])
            border = float(split['border'])

            idx = next(split_idx)
            return_tree.add_split(idx, '>', feature_idx, border)

            if parent_idx is not None:
                return_tree.add_edge(parent_idx, idx, is_child_leaf=False)

            traverse(li[1::2], parent_idx=idx, it=it+1)
            traverse(li[0::2], parent_idx=idx, it=it+1)

        traverse(leaf_indices)

        return return_tree

    def predict(self, data: np.ndarray) -> np.ndarray:
        root_idx = self.node_name(0)

        def recurse(X, node_idx):
            node = self.tree.nodes[node_idx]

            threshold = node['threshold']
            feature = node['feature']

            left_child_idx, right_child_idx = list(
                self.tree.successors(node_idx))

            if X[feature] > threshold:
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
