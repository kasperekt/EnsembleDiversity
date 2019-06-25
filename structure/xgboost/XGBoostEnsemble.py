import numpy as np
import xgboost as xgb
import scipy
import scipy.special as sp

from data import Dataset, DatasetEncoder
from structure import Ensemble
from .XGBoostTree import XGBoostTree


class XGBoostEnsemble(Ensemble):
    def __init__(self, params):
        super().__init__(params, name='XGBoostEnsemble')
        self.clf = xgb.XGBClassifier(**params)
        self.dataset_encoder = None

    def fit(self, dataset: Dataset):
        if self.dataset_encoder is None:
            self.dataset_encoder = DatasetEncoder.create_one_hot(dataset)

        encoded_dataset = self.dataset_encoder.transform(dataset)

        self.trees = []
        self.clf.fit(encoded_dataset.X, encoded_dataset.y)

        n_classes = encoded_dataset.num_classes()
        n_estimators = self.clf.n_estimators
        n_trees = n_estimators * n_classes if n_classes > 2 else n_estimators

        for tree_idx in range(0, n_trees):
            tree = xgb.to_graphviz(self.clf, num_trees=tree_idx)
            parsed_tree = XGBoostTree.parse(str(tree), encoded_dataset)

            self.trees.append(parsed_tree)

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        if self.dataset_encoder is None:
            raise ValueError('No dataset encoder available')

        encoded_dataset = self.dataset_encoder.transform(dataset)

        n_classes = len(self.clf.classes_)
        n_estimators = self.clf.n_estimators
        n_examples = len(encoded_dataset.X)

        predictions = np.array([tree.predict(encoded_dataset.X)
                                for tree in self.trees])

        if n_classes > 2:
            predictions = np.rollaxis(predictions, axis=1).reshape(
                (n_examples, n_estimators, n_classes))
            probs = sp.softmax(np.sum(predictions, axis=1), axis=1)
        else:
            activated = sp.expit(np.sum(predictions, axis=0))
            probs = np.array([np.array([1 - prob, prob])
                              for prob in activated])

        return probs

    def predict(self, dataset: Dataset) -> np.ndarray:
        results_proba = self.predict_proba(dataset)
        results_cls = np.argmax(results_proba, axis=1)

        return results_cls
