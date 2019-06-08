import numpy as np

from .Experiment import Experiment
from typing import List
from structure import Dataset, XGBoostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class XGBoostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = XGBoostEnsemble
        self.name = 'XGBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 300, 30),
            'max_depth': range(2, 8),
            'num_leaves': range(30, 50, 3),
            'learning_rate': [0.1]
        })
