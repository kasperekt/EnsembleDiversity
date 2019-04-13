import networkx as nx
import pygraphviz as pgv

from typing import List


class TreeStructure:
    def __init__(self, feature_names: List[str], target_names: List[str]):
        self.tree = nx.DiGraph()
        self.feature_names = feature_names
        self.target_names = target_names

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

    def add_leaf(self, idx: int, value: float, target: int = -1, count: int = 0):
        self.tree.add_node(self.leaf_name(idx),
                           value=value,
                           count=count,
                           target=target,
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

    def draw(self, path: str):
        tree_copy = self.tree.copy()
        for node_idx, node_data in tree_copy.nodes(data=True):
            if node_data['is_split']:
                feature = self.feature_names[node_data['feature']]
                decision_type = node_data['decision_type']
                threshold = node_data['threshold']
                node_data['label'] = f'{feature}\n{decision_type}\n{threshold}'
            elif node_data['is_leaf']:
                target = node_data['target']
                target_name = self.target_names[target] if target != -1 else 'n/d'
                node_data['label'] = f"{target_name}\n{node_data['value']}"

        graph_str = str(nx.nx_agraph.to_agraph(tree_copy))
        pgv_graph = pgv.AGraph(graph_str)
        pgv_graph.layout(prog='dot')
        pgv_graph.draw(path)

    def __repr__(self):
        pgv_graph = nx.nx_agraph.to_agraph(self.tree)
        return str(pgv_graph)
