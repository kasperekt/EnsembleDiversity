from .Experiment import Experiment
from typing import List
from structure import Dataset, RandomForestEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class RandomForestExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.name = 'RandomForestExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': [10, 20, 50, 100],
            'max_depth': range(2, 30, 2),
        })

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:
                ensemble = RandomForestEnsemble(params)
                ensemble.fit(train)

                preds = ensemble.predict(val.X)
                accuracy = accuracy_score(val.y, preds)

                node_diversity = ensemble.node_diversity()

                result_dict = {
                    'name': ensemble.name,
                    'dataset_name': train.name,
                    'accuracy': accuracy,
                    'node_diversity': node_diversity,
                    **params
                }

                self.add_result(**result_dict)
                self.ensembles.append(ensemble)
