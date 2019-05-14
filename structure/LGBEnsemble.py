import numpy as np
import lightgbm as lgb
import scipy.special as sp

from .Ensemble import Ensemble
from .Dataset import Dataset
from .LGBTree import LGBTree


class LGBEnsemble(Ensemble):
    def __init__(self, params: dict, name='LightGBM'):
        super().__init__(params, name)
        self.clf = lgb.LGBMClassifier(**params)

    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)

        json_object = self.clf.booster_.dump_model()
        trees = json_object['tree_info']

        self.trees = [LGBTree.parse(tree, dataset) for tree in trees]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        n_classes = len(self.clf.classes_)
        n_estimators = self.clf.n_estimators

        predictions = np.array([tree.predict(X) for tree in self.trees])

        if self.clf.objective_ == 'multiclass':
            predictions = np.rollaxis(predictions, axis=1).reshape(
                (len(X), n_estimators, n_classes))
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
