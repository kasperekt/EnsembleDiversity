import numpy as np
import xgboost as xgb
import scipy
import scipy.special as sp

from structure import Ensemble, Dataset
from .XGBoostTree import XGBoostTree


class XGBoostEnsemble(Ensemble):
    def __init__(self, params):
        super().__init__(params, name='XGBoostEnsemble')
        self.clf = xgb.XGBClassifier(**params)

    def fit(self, dataset: Dataset):
        self.trees = []
        self.clf.fit(dataset.X, dataset.y)

        n_classes = dataset.num_classes()
        n_estimators = self.clf.n_estimators
        n_trees = n_estimators * n_classes if n_classes > 2 else n_estimators

        for tree_idx in range(0, n_trees):
            tree = xgb.to_graphviz(self.clf, num_trees=tree_idx)
            parsed_tree = XGBoostTree.parse(str(tree), dataset)

            self.trees.append(parsed_tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        n_classes = len(self.clf.classes_)
        n_estimators = self.clf.n_estimators

        predictions = np.array([tree.predict(X) for tree in self.trees])

        if n_classes > 2:
            predictions = np.rollaxis(predictions, axis=1).reshape(
                (len(X), n_estimators, n_classes))
            probs = sp.softmax(np.sum(predictions, axis=1), axis=1)
        else:
            activated = sp.expit(np.sum(predictions, axis=0))
            probs = np.array([np.array([1 - prob, prob])
                              for prob in activated])

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        results_proba = self.predict_proba(X)
        results_cls = np.argmax(results_proba, axis=1)

        return results_cls
