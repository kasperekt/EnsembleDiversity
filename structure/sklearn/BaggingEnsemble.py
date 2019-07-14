from .SklearnEnsemble import SklearnEnsemble

from data import Dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


class BaggingEnsemble(SklearnEnsemble):
    def __init__(self, params: dict, dataset: Dataset = None):
        # Always use all cores
        params['n_jobs'] = -1

        super().__init__(params, dataset, name='Bagging')

        clf_params = params.copy()
        del clf_params['max_depth']

        tree_params = {'max_depth': params['max_depth']}

        tree = DecisionTreeClassifier(**tree_params)
        self.clf = BaggingClassifier(base_estimator=tree, **clf_params)
