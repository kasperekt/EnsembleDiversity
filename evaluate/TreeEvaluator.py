import numpy as np

from typing import List
from structure.Tree import Tree


class TreeEvaluator:
    def __init__(self, trees: List[Tree]):
        self.trees = trees

    def majority_voting(self):
        pass

    def evaluate(self, X, voting_type='majority'):
        preds = np.array([tree.predict(X) for tree in self.trees])

        if voting_type == 'majority':
            results = self.majority_voting(preds)
        else:
            raise ValueError('Invalid voting_type passed. {}'.format(voting_type))
