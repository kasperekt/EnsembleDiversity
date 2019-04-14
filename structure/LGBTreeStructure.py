from .TreeStructure import TreeStructure


class LGBTreeStructure(TreeStructure):
    def __init__(self, feature_names, target_names):
        super().__init__(feature_names, target_names, clf_type="LightGBM")

    def add_leaf(self, idx, value=0.0):
        self.tree.add_node(self.leaf_name(idx), value=value, is_leaf=True, is_split=False)

    def leaf_label(self, node_data: dict) -> str:
        return f"{node_data['value']}"
