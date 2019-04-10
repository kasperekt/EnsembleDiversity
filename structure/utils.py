def is_leaf(structure: dict) -> bool:
    return 'leaf_index' in structure


def is_split(structure: dict) -> bool:
    return 'split_index' in structure
