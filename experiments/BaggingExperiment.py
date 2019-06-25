from .Experiment import Experiment
from typing import List
from data import Dataset
from structure import BaggingEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class BaggingExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = BaggingEnsemble

        self.name = 'BaggingExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 30),
            'max_depth': range(2, 30, 2),
        })
