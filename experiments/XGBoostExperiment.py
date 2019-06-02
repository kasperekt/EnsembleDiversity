import numpy as np

from .Experiment import Experiment
from typing import List
from structure import Dataset, XGBoostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class XGBoostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.name = 'XGBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 10),
            'max_depth': range(2, 15),
            'num_leaves': range(30, 50, 3),
            'learning_rate': [0.1]
        })

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:
                final_params = {**params}

                ensemble = XGBoostEnsemble(final_params)
                ensemble.fit(train)

                preds = ensemble.predict(val.X)
                accuracy = accuracy_score(val.y, preds)

                node_diversity = ensemble.node_diversity()

                result_dict = {
                    'name': ensemble.name,
                    'dataset_name': train.name,
                    'accuracy': accuracy,
                    'node_diversity': node_diversity,
                    **final_params
                }

                self.add_result(**result_dict)
                self.ensembles.append(ensemble)
