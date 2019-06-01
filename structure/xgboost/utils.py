import pygraphviz as pgv


def build_edges_dict(graph: pgv.AGraph) -> dict:
    edges = {}

    for src_node, dest_node in graph.edges():
        if src_node not in edges:
            edges[src_node] = []

        edges[src_node].append(dest_node)

    return edges


def get_leaf_value(node_data: dict) -> float:
    label = node_data['label']
    return float(label.split('=')[1])


def get_feature(node_data: dict) -> int:
    # Example: f10<0.15
    label = node_data['label']
    feature_str = label.split('<')[0]
    return int(feature_str[1:])


def get_threshold(node_data: dict):
    # Example: f11<0.14
    label = node_data['label']
    threshold_str = label.split('<')[1]
    return float(threshold_str)


def is_leaf(node_data: dict) -> bool:
    label = node_data['label']
    return 'leaf' in label
