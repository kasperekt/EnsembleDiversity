from .SklearnEnsemble import SklearnEnsemble

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


class BaggingEnsemble(SklearnEnsemble):
    def __init__(self, params: dict):
        super().__init__(params, 'Bagging')

        clf_params = params.copy()
        del clf_params['max_depth']

        tree_params = {'max_depth': params['max_depth']}

        tree = DecisionTreeClassifier(**tree_params)
        self.clf = BaggingClassifier(base_estimator=tree, **clf_params)
