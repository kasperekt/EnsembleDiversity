import numpy as np
import lightgbm as lgb
import scipy.special as sp

from .LGBTree import LGBTree
from data import Dataset
from structure import Ensemble, Tree


class LGBEnsemble(Ensemble):
    def __init__(self, params: dict, dataset: Dataset = None, name='LightGBM'):
        # Always use all cores
        params['n_jobs'] = -1

        super().__init__(params, dataset, name)
        self.clf = lgb.LGBMClassifier(**params)

    def fit(self, dataset: Dataset):
        self.set_dataset(dataset)

        self.clf.fit(self.dataset.X, self.dataset.y)

        json_object = self.clf.booster_.dump_model()
        trees = json_object['tree_info']

        self.trees = [LGBTree.parse(tree, self.dataset) for tree in trees]

    def predict(self, dataset: Dataset) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        encoded_dataset = self.encode_dataset(dataset)

        n_classes = len(self.clf.classes_)
        n_estimators = self.clf.n_estimators
        n_examples = len(encoded_dataset.X)

        predictions = np.array([tree.predict(encoded_dataset.X)
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
