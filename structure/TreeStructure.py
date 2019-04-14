import networkx as nx
import pygraphviz as pgv

from abc import abstractmethod


class TreeStructure(object):
    def __init__(self, dataset, clf_type="n/d"):
        self.tree = nx.DiGraph()
        self.dataset = dataset
        self.clf_type = clf_type

    @staticmethod
    def node_name(idx: int):
        return f'Node_{idx}'

    @staticmethod
    def leaf_name(idx: int):
        return f'Leaf_{idx}'

    def add_split(self, idx: int, decision_type: str, feature: str, threshold: float):
        self.tree.add_node(self.node_name(idx),
                           decision_type=decision_type,
                           feature=feature,
                           threshold=threshold,
                           is_split=True,
                           is_leaf=False)

    @abstractmethod
    def add_leaf(self, idx: int, **kwargs):
        self.tree.add_node(self.leaf_name(idx),
                           **kwargs,
                           is_split=False,
                           is_leaf=True)

    def add_edge(self, parent_idx: int, child_idx: int, is_child_leaf=False):
        parent_name = self.node_name(parent_idx)
        child_name = self.leaf_name(child_idx) if is_child_leaf else self.node_name(child_idx)

        self.tree.add_edge(parent_name,
                           child_name,
                           threshold='HARDCODED')

    def num_nodes(self):
        return len(self.tree.nodes)

    def num_edges(self):
        return len(self.tree.edges)

    @abstractmethod
    def leaf_label(self, node_data: dict):
        raise NotImplementedError('"leaf_label" method is not implemented')

    def draw(self, path: str):
        tree_copy = self.tree.copy()
        for node_idx, node_data in tree_copy.nodes(data=True):
            if node_data['is_split']:
                feature = self.dataset.feature_names[node_data['feature']]
                decision_type = node_data['decision_type']
                threshold = node_data['threshold']
                node_data['label'] = f'{feature}\n{decision_type}\n{threshold}'
            elif node_data['is_leaf']:
                node_data['label'] = self.leaf_label(node_data)

        graph_str = str(nx.nx_agraph.to_agraph(tree_copy))
        pgv_graph = pgv.AGraph(graph_str)
        pgv_graph.layout(prog='dot')
        pgv_graph.draw(path)

    def __repr__(self):
        pgv_graph = nx.nx_agraph.to_agraph(self.tree)
        return str(pgv_graph)
