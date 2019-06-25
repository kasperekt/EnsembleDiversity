import numpy as np
import lightgbm as lgb
import scipy.special as sp

from .LGBTree import LGBTree
from data import Dataset
from structure import Ensemble, Tree


class LGBEnsemble(Ensemble):
    def __init__(self, params: dict, name='LightGBM'):
        super().__init__(params, name)
        self.clf = lgb.LGBMClassifier(**params)

    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)

        json_object = self.clf.booster_.dump_model()
        trees = json_object['tree_info']

        self.trees = [LGBTree.parse(tree, dataset) for tree in trees]

    def predict(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        n_classes = len(self.clf.classes_)
        n_estimators = self.clf.n_estimators
        n_examples = len(dataset.X)

        predictions = np.array([tree.predict(dataset.X)
                                for tree in self.trees])

        if self.clf.objective_ == 'multiclass':
            predictions = np.rollaxis(predictions, axis=1).reshape(
                (n_examples, n_estimators, n_classes))
            probs = sp.softmax(np.sum(predictions, axis=1), axis=1)
        elif self.clf.objective_ == 'binary':
            # Expit is just sigmoid
            # pylint: disable=no-member
            activated = sp.expit(np.sum(predictions, axis=0))
            probs = np.array([np.array([1 - prob, prob])
                              for prob in activated])
        else:
            raise NotImplementedError()

        return np.argmax(probs, axis=1)
