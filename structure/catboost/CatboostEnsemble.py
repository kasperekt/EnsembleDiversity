import json
import numpy as np

from .CatboostTree import CatboostTree
from data import Dataset
from structure import Ensemble
from catboost import CatBoostClassifier
from scipy.special import expit, softmax  # pylint: disable=no-name-in-module


class CatboostEnsemble(Ensemble):
    def __init__(self, params: dict, dataset: Dataset = None):
        super().__init__(params, dataset, name='CatboostEnsemble')
        self.clf = CatBoostClassifier(**params)
        self.tmp_json_path = '/tmp/catboost.model.json'

    def fit(self, dataset: Dataset):
        self.set_dataset(dataset)

        loss_function = 'MultiClass' if self.dataset.num_classes() > 2 else 'Logloss'
        self.clf.set_params(loss_function=loss_function, verbose=False)

        self.clf.fit(self.dataset.X, self.dataset.y)

        self.clf.save_model(self.tmp_json_path, format='json')
        with open(self.tmp_json_path, 'r') as fp:
            model = json.load(fp)

        self.trees = [CatboostTree.parse(tree, self.dataset)
                      for tree in model['oblivious_trees']]

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        encoded_dataset = self.encode_dataset(dataset)

        n_classes = len(self.clf.classes_)  # pylint: disable=no-member

        # TODO: For single tree this is just [tree.predict(...)]
        preds = np.array([tree.predict(encoded_dataset.X)
                          for tree in self.trees])
        preds = np.sum(preds, axis=0)

        if n_classes > 2:
            # https://catboost.ai/docs/concepts/loss-functions-multiclassification.html
            # Link above suggests different equation for this
            results_proba = softmax(preds, axis=1)
        else:
            # results_proba = np.array([[1 - v, v] for v in expit(preds)])
            raise NotImplementedError('Only binary problems are implemented.')

        return results_proba

    def predict(self, dataset: Dataset) -> np.ndarray:
        results_proba = self.predict_proba(dataset)
        results_cls = np.argmax(results_proba, axis=1)
        return results_cls
