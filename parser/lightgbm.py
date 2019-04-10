from typing import List

from structure.TreeStructure import TreeStructure
from structure.utils import is_split, is_leaf


def node_index_of(child):
    if is_leaf(child):
        return child['leaf_index']
    elif is_split(child):
        return child['split_index']
    else:
        raise ValueError('Child is not a valid structure')


def parse_tree(tree, feature_names):
    return_tree = TreeStructure(feature_names)

    def traverse(structure):
        if is_leaf(structure):
            return_tree.add_leaf(structure['leaf_index'],
                                 structure['leaf_value'],
                                 structure['leaf_count'])
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

            return_tree.add_edge(structure, left_child)
            return_tree.add_edge(structure, right_child)

    traverse(tree['tree_structure'])

    return return_tree


def parse_lightgbm_json(json_object) -> List[TreeStructure]:
    feature_names = json_object['feature_names']
    trees = json_object['tree_info']

    structures = [parse_tree(tree, feature_names) for tree in trees]
    
    return structures
