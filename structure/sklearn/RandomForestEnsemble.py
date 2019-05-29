from . import SklearnEnsemble
from sklearn.ensemble import RandomForestClassifier


class RandomForestEnsemble(SklearnEnsemble):
    def __init__(self, params, name='RandomForest'):
        super().__init__(params, name='RandomForest')

        self.clf = RandomForestClassifier(**params)
