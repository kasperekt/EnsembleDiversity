import numpy as np

from . import SklearnEnsemble, SklearnTree
from structure import Dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaboostEnsemble(SklearnEnsemble):
    def __init__(self, params, name='Adabost'):
        super().__init__(params, name='Adaboost')

        clf_params = {**params}
        del clf_params['max_depth']

        tree_params = {'max_depth': params['max_depth']}

        tree = DecisionTreeClassifier(**tree_params)
        self.clf = AdaBoostClassifier(
            **clf_params, base_estimator=tree, algorithm='SAMME')

    def predict(self, X: np.ndarray) -> np.ndarray:
        weights = self.clf.estimator_weights_
        classes = self.clf.classes_
        n_classes = len(classes)

        classes = classes[:, np.newaxis]
        preds = np.array([(tree.predict(X) == classes).T *
                          w for tree, w in zip(self.trees, weights)])
        preds = sum(preds)
        preds /= self.clf.estimator_weights_.sum()

        if n_classes == 2:
            preds[:, 0] *= -1
            preds = preds.sum(axis=1)
            return self.clf.classes_.take(preds > 0, axis=0)

        preds = np.argmax(preds, axis=1)
        return self.clf.classes_.take(preds, axis=0)
