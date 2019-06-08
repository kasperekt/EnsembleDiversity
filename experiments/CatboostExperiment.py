import numpy as np

from .Experiment import Experiment
from typing import List
from structure import Dataset, CatboostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class CatboostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = CatboostEnsemble

        self.name = 'CatboostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 300, 30),
            'max_depth': range(2, 8),
            'learning_rate': [0.1]
        })
