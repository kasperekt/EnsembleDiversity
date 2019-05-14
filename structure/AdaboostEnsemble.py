from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from .SklearnEnsemble import SklearnEnsemble
from .SklearnTree import SklearnTree
from .Dataset import Dataset


class AdaboostEnsemble(SklearnEnsemble):
    def __init__(self, params, name='Adabost'):
        super().__init__(params, name='Adaboost')

        clf_params = {**params}
        del clf_params['max_depth']

        tree_params = {'max_depth': params['max_depth']}

        tree = DecisionTreeClassifier(**tree_params)
        self.clf = AdaBoostClassifier(**clf_params, base_estimator=tree)
