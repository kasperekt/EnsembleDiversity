from .RankExperiment import RankExperiment, ExperimentVariant
from typing import List
from data import Dataset
from structure import AdaboostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class AdaboostExperiment(RankExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.EnsembleType = AdaboostEnsemble

        self.name = 'AdaBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 30),
            'max_depth': range(2, 30, 2),
            'learning_rate': [0.1]
        })
