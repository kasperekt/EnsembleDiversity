import networkx as nx
import pygraphviz as pgv

from structure.utils import is_leaf


class TreeStructure:
    def __init__(self, feature_names):
        self.tree = nx.DiGraph()
        self.feature_names = feature_names

    @staticmethod
    def node_name(idx):
        return f'Node_{idx}'

    @staticmethod
    def leaf_name(idx):
        return f'Leaf_{idx}'

    def add_split(self, idx, decision_type, feature, threshold):
        self.tree.add_node(self.node_name(idx),
                           decision_type=decision_type,
                           feature=feature,
                           threshold=threshold,
                           is_split=True,
                           is_leaf=False)

    def add_leaf(self, idx, value, count):
        self.tree.add_node(self.leaf_name(idx),
                           value=value,
                           count=count,
                           is_split=False,
                           is_leaf=True)

    def add_edge(self, parent, child):
        parent_idx = self.node_name(parent['split_index'])
        child_idx = self.leaf_name(child['leaf_index']) if is_leaf(child) else self.node_name(child['split_index'])
        self.tree.add_edge(parent_idx, child_idx, threshold='HARDCODED')

    def num_nodes(self):
        return len(self.tree.nodes)

    def num_edges(self):
        return len(self.tree.edges)

    def draw(self, path):
        tree_copy = self.tree.copy()
        for node_idx, node_data in tree_copy.nodes(data=True):
            if node_data['is_split']:
                feature = self.feature_names[node_data['feature']]
                decision_type = node_data['decision_type']
                threshold = node_data['threshold']
                node_data['label'] = f'{feature}\n{decision_type}\n{threshold}'
            elif node_data['is_leaf']:
                node_data['label'] = node_data['value']

        graph_str = str(nx.nx_agraph.to_agraph(tree_copy))
        pgv_graph = pgv.AGraph(graph_str)
        pgv_graph.layout(prog='dot')
        pgv_graph.draw(path)

    def __repr__(self):
        pgv_graph = nx.nx_agraph.to_agraph(self.tree)
        return str(pgv_graph)
