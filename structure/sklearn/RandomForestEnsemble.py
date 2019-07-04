from . import SklearnEnsemble
from sklearn.ensemble import RandomForestClassifier


class RandomForestEnsemble(SklearnEnsemble):
    def __init__(self, params):
        # Always use all cores
        params['n_jobs'] = -1

        super().__init__(params, name='RandomForest')
        self.clf = RandomForestClassifier(**params)
