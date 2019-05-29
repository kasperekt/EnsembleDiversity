import json
import numpy as np

from . import CatboostTree
from structure import Dataset, Ensemble
from catboost import CatBoostClassifier
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module


class CatboostEnsemble(Ensemble):
    def __init__(self, params, name='CatboostEnsemble'):
        super().__init__(params, name=name)
        self.clf = CatBoostClassifier(**params)
        self.tmp_json_path = '/tmp/catboost.model.json'

    def fit(self, dataset: Dataset):
        # self.clf.set_params() - pick objective??
        self.clf.fit(dataset.X, dataset.y)

        self.clf.save_model(self.tmp_json_path, format='json')
        with open(self.tmp_json_path, 'r') as fp:
            model = json.load(fp)

        self.trees = [CatboostTree.parse(tree, dataset)
                      for tree in model['oblivious_trees']]

    def predict_proba(self, X) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        n_classes = len(self.clf._classes)  # pylint: disable=no-member

        preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.sum(preds, axis=0)

        if n_classes > 2:
            results_proba = softmax(preds, axis=1)
        else:
            print(preds)
            results_proba = np.array([[1 - v, v] for v in expit(preds)])

        return results_proba

    def predict(self, X) -> np.ndarray:
        results_proba = self.predict_proba(X)
        results_cls = np.argmax(results_proba, axis=1)
        return results_cls
