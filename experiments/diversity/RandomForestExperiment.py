from .DiversityExperiment import DiversityExperiment, ExperimentVariant
from typing import List
from data import Dataset
from structure import RandomForestEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class RandomForestExperiment(DiversityExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.EnsembleType = RandomForestEnsemble

        self.name = 'RandomForestExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 20),
            'max_depth': range(2, 30, 3),
        })
