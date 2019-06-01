import numpy as np
import pygraphviz as pgv
import networkx as nx

from .utils import build_edges_dict, is_leaf, get_leaf_value, get_threshold, get_feature
from structure import LeafValueTree, Dataset


class XGBoostTree(LeafValueTree):
    def __init__(self, dataset):
        super().__init__(dataset, clf_type='xgboost')

    @staticmethod
    def parse(tree_dot: str, dataset: Dataset):
        return_tree = XGBoostTree(dataset)

        graph = nx.drawing.nx_agraph.from_agraph(pgv.AGraph(tree_dot))
        edges_dict = build_edges_dict(graph)

        for node_idx, node_data in graph.nodes(data=True):
            if is_leaf(node_data):
                return_tree.add_leaf(node_idx,
                                     value=get_leaf_value(node_data))
            else:
                return_tree.add_split(node_idx, '<',
                                      get_feature(node_data),
                                      get_threshold(node_data))

                successors = edges_dict[node_idx]

                for child_idx in successors:
                    child_node = graph.nodes[child_idx]
                    return_tree.add_edge(
                        node_idx, child_idx, is_child_leaf=is_leaf(child_node))

        return return_tree
