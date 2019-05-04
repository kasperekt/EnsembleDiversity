import numpy as np
import lightgbm as lgb

from scipy.special import softmax
from .Ensemble import Ensemble
from .Dataset import Dataset
from .LGBTree import LGBTree


class LGBEnsemble(Ensemble):
    def __init__(self, clf: lgb.LGBMClassifier, name='Sklearn'):
        super().__init__(clf, name)
    
    def fit(self, dataset: Dataset):
        self.clf.fit(dataset.X, dataset.y)

        json_object: dict = self.clf.booster_.dump_model()
        trees = json_object['tree_info']

        self.trees = [LGBTree.parse(tree, dataset) for tree in trees]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.trees) == 0:
            raise ValueError('There are no trees available')

        predictions = np.array([tree.predict(X) for tree in self.trees])

        if self.clf.objective_ == 'multiclass':
            n_classes = len(self.clf.classes_)
            n_estimators = len(self.trees) // n_classes
            predictions = np.rollaxis(predictions, axis=1).reshape((len(X), n_estimators, n_classes))
            probs = softmax(np.sum(predictions, axis=1), axis=1)
        else:
            raise NotImplementedError('Only "multiclass" logic')

        return np.argmax(probs, axis=1)
