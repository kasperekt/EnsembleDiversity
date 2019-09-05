import numpy as np

from .DiversityExperiment import DiversityExperiment, ExperimentVariant
from typing import List
from data import Dataset
from structure import XGBoostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class XGBoostExperiment(DiversityExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.EnsembleType = XGBoostEnsemble
        self.name = 'XGBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 20),
            'max_depth': range(2, 10),
            'num_leaves': range(30, 50, 3),
            'learning_rate': [0.1]
        })
