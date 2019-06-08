from .Experiment import Experiment
from typing import List
from structure import Dataset, AdaboostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class AdaboostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = AdaboostEnsemble

        self.name = 'AdaBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': [10, 20, 50, 100],
            'max_depth': range(2, 20, 2),
            'learning_rate': [0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
        })
