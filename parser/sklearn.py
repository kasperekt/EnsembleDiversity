import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.ensemble import AdaBoostClassifier
from structure.TreeStructure import TreeStructure
from typing import List


def parse_tree(decision_tree: DecisionTreeClassifier, feature_names: List[str], target_names: List[str]):
    '''
    What does "-2", "-1" mean for thresholds, etc.
    '''
    tree_structure = TreeStructure(feature_names, target_names)

    children_left = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    threshold = decision_tree.tree_.threshold
    # features = [feature_names[i] for i in decision_tree.tree_.feature]
    features = decision_tree.tree_.feature
    value = decision_tree.tree_.value

    def traverse(node: int):
        if threshold[node] != -2:
            tree_structure.add_split(node,
                                     decision_type='<=',
                                     feature=features[node],
                                     threshold=threshold[node])

            if children_left[node] != -1:
                left_child = children_left[node]
                traverse(left_child)
                tree_structure.add_edge(node,
                                        left_child,
                                        is_child_leaf=threshold[left_child] == -2)

            if children_right[node] != -1:
                right_child = children_right[node]
                traverse(right_child)
                tree_structure.add_edge(node,
                                        right_child,
                                        is_child_leaf=threshold[right_child] == -2)
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_fraction = v
                # TODO: last param in add_leaf should be count
                tree_structure.add_leaf(node, target_fraction, target=i)

    traverse(0)

    return tree_structure


# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
# https://towardsdatascience.com/uncover-the-structure-of-tree-ensembles-in-python-a01f72ea54a2
# https://www.garysieling.com/blog/convert-scikit-learn-decision-trees-json
# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
def parse_sklearn(clf: AdaBoostClassifier, feature_names: List[str], target_names: List[str]):
    structures = []

    for tree in clf.estimators_:
        tree_structure = parse_tree(tree, feature_names, target_names)
        structures.append(tree_structure)

    return structures
