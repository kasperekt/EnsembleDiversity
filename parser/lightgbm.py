from lightgbm import LGBMClassifier
from structure.LGBTreeStructure import LGBTreeStructure
from structure.DatasetStructure import DatasetStructure
from structure.utils import is_split, is_leaf


def node_index_of(child: dict):
    if is_leaf(child):
        return child['leaf_index']
    elif is_split(child):
        return child['split_index']
    else:
        raise ValueError('Child is not a valid structure')


def parse_tree(tree: dict, dataset: DatasetStructure) -> LGBTreeStructure:
    return_tree = LGBTreeStructure(dataset)

    def traverse(structure):
        if is_leaf(structure):
            return_tree.add_leaf(structure['leaf_index'],
                                 structure['leaf_value'])
        elif is_split(structure):
            split_index = structure['split_index']
            decision_type = structure['decision_type']
            split_feature = structure['split_feature']
            threshold = structure['threshold']

            return_tree.add_split(split_index, decision_type, split_feature, threshold)

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


def parse_lightgbm(clf: LGBMClassifier, dataset: DatasetStructure):
    json_object: dict = clf.booster_.dump_model()
    trees = json_object['tree_info']

    structures = [parse_tree(tree, dataset) for tree in trees]

    return structures
