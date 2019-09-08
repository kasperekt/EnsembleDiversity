from . import SklearnEnsemble
from sklearn.ensemble import RandomForestClassifier
from data import Dataset


class RandomForestEnsemble(SklearnEnsemble):
    def __init__(self, params, dataset: Dataset = None):
        super().__init__(params, dataset, name='RandomForest')
        self.clf = RandomForestClassifier(**params)
