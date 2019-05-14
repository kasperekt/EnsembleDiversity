from .Experiment import Experiment
from typing import List
from structure import Dataset, AdaboostEnsemble
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class AdaboostExperiment(Experiment):
    def __init__(self):
        super().__init__()

        self.name = 'AdaBoostExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': [10, 20, 50, 100],
            'max_depth': range(2, 20, 2),
            'learning_rate': [0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]
        })

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:
                ensemble = AdaboostEnsemble(params)
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
