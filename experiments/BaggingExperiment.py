from .Experiment import Experiment
from typing import List
from structure import Dataset, BaggingEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class BaggingExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = BaggingEnsemble

        self.name = 'BaggingExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': [10, 20, 50, 100, 200, 300, 500],
            'max_depth': range(2, 30, 2),
        })
