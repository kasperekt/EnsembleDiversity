import lightgbm as lgb

from typing import List
from .DiversityExperiment import DiversityExperiment, ExperimentVariant
from data import Dataset
from structure import LGBEnsemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid


class LGBExperiment(DiversityExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.EnsembleType = LGBEnsemble

        self.name = 'LGBExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 30),
            'max_depth': range(2, 8),
            'num_leaves': range(20, 50, 3),
            'learning_rate': [0.1]
        })

    def pick_objective(self, dataset: Dataset) -> str:
        if dataset.num_classes() > 2:
            return 'multiclass'

        return 'binary'

    def get_final_params(self, params, train, val):
        return {**params, 'objective': self.pick_objective(train)}
