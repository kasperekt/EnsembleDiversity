def is_leaf(structure: dict) -> bool:
    return 'leaf_index' in structure


def is_split(structure: dict) -> bool:
    return 'split_index' in structure


def node_index_of(child: dict):
    if is_leaf(child):
        return child['leaf_index']
    elif is_split(child):
        return child['split_index']
