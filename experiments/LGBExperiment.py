import lightgbm as lgb

from typing import List
from .Experiment import Experiment
from structure import LGBEnsemble, Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid


class LGBExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.EnsembleType = LGBEnsemble

        self.name = 'LGBExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 10),
            'max_depth': range(2, 15),
            'num_leaves': range(30, 50, 3),
            'learning_rate': [0.1]
        })

    def pick_objective(self, dataset: Dataset) -> str:
        if dataset.num_classes() > 2:
            return 'multiclass'

        return 'binary'

    def get_final_params(self, params, train, val):
        return {**params, 'objective': self.pick_objective(train)}
