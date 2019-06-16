import networkx as nx
import pygraphviz as pgv
import numpy as np

from abc import abstractmethod


class Tree(object):
    def __init__(self, dataset, clf_type="n/d"):
        self.tree = nx.DiGraph()
        self.dataset = dataset
        self.clf_type = clf_type

        self.used_features = set()
        self.used_attrs = set()

    @staticmethod
    def node_name(idx: int):
        return f'Node_{idx}'

    @staticmethod
    def leaf_name(idx: int):
        return f'Leaf_{idx}'

    @abstractmethod
    def leaf_label(self, node_data: dict):
        raise NotImplementedError('"leaf_label" method is not implemented')

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError('"predict" method not implemented.')

    @abstractmethod
    def add_leaf(self, idx: int, **kwargs):
        self.tree.add_node(self.leaf_name(idx),
                           **kwargs,
                           is_leaf=True)

    def add_split(self, idx: int, decision_type: str, feature: int, threshold: float):
        self.add_used_attr(feature, threshold, decision_type)
        self.add_used_feature(feature)

        self.tree.add_node(self.node_name(idx),
                           decision_type=decision_type,
                           feature=feature,
                           threshold=threshold,
                           is_leaf=False)

    def add_edge(self, parent_idx: int, child_idx: int, is_child_leaf=False):
        parent_name = self.node_name(parent_idx)
        child_name = self.leaf_name(
            child_idx) if is_child_leaf else self.node_name(child_idx)

        threshold = self.tree.nodes[parent_name]['threshold']

        self.tree.add_edge(parent_name, child_name, threshold=threshold)

    def edge_label(self):
        return

    def encode_used_attr(self, feature, threshold, decision_type):
        return f'{feature}__{threshold}__${decision_type}'

    def add_used_attr(self, feature, threshold, decision_type):
        used_attr_encoded = self.encode_used_attr(
            feature, threshold, decision_type)
        self.used_attrs.add(used_attr_encoded)

    def add_used_feature(self, feature):
        self.used_features.add(feature)

    def num_nodes(self) -> int:
        return len(self.tree.nodes)

    def num_edges(self):
        return len(self.tree.edges)

    def satisfies_cond(self, cond_type, lvalue, rvalue):
        if cond_type == '>':
            return lvalue > rvalue
        if cond_type == '>=':
            return lvalue >= rvalue
        if cond_type == '<':
            return lvalue < rvalue
        if cond_type == '<=':
            return lvalue <= rvalue

        raise ValueError(f'{cond_type} is not implemented.')

    def draw(self, path: str):
        tree_copy = self.tree.copy()

        for _, node_data in tree_copy.nodes(data=True):
            if node_data['is_leaf']:
                node_data['label'] = self.leaf_label(node_data)
            else:
                feature = self.dataset.feature_names[node_data['feature']]
                decision_type = node_data['decision_type']
                threshold = node_data['threshold']
                node_data['label'] = f'{feature}\n{decision_type}\n{threshold}'

        graph_str = str(nx.nx_agraph.to_agraph(tree_copy))
        pgv_graph = pgv.AGraph(graph_str)
        pgv_graph.layout(prog='dot')
        pgv_graph.draw(path)

    def to_str(self):
        pgv_graph = nx.nx_agraph.to_agraph(self.tree)
        return str(pgv_graph)

    def __repr__(self):
        return f'Tree<nodes_len={len(self.tree.nodes)}>'
