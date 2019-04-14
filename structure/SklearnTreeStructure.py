from .TreeStructure import TreeStructure
from .DatasetStructure import DatasetStructure


class SklearnTreeStructure(TreeStructure):
    def __init__(self, dataset: DatasetStructure):
        super().__init__(dataset, clf_type="Sklearn")

    def add_leaf(self, idx, target=-1, fraction=0.0):
        self.tree.add_node(self.leaf_name(idx), target=target, fraction=fraction, is_leaf=True, is_split=False)

    def leaf_label(self, node_data: dict) -> str:
        target = node_data['target']
        target_name = self.dataset.target_names[target] if target != -1 else 'n/d'
        return f"{target_name}\n{node_data['fraction']}"
