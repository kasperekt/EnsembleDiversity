import numpy as np

from .Experiment import Experiment
from typing import List
from structure import Dataset, CatboostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class CatboostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.name = 'CatboostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': range(10, 500, 10),
            'max_depth': range(2, 15),
            'learning_rate': [0.1]
        })

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:
                final_params = {**params}

                ensemble = CatboostEnsemble(final_params)
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
