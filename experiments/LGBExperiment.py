import lightgbm as lgb

from typing import List
from .Experiment import Experiment
from structure import LGBEnsemble, Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid


class LGBExperiment(Experiment):
    def __init__(self, datasets):
        super().__init__(datasets)

        self.name = 'LGBExperiment'
        self.param_grid = ParameterGrid({
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [10, 20, 30]
        })

    def pick_objective(self, dataset: Dataset) -> str:
        if dataset.num_classes() > 2:
            return 'multiclass'

        return 'binary'

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for idx, (train_dataset, val_dataset) in enumerate(zip(train_data, val_data)):
            for params in self.param_grid:
                final_params = {**params,
                                'objective': self.pick_objective(train_dataset)}

                ensemble = LGBEnsemble(
                    final_params, f'LGBEnsemble_{idx}_{train_dataset.name}')
                ensemble.fit(train_dataset)

                preds = ensemble.predict(val_dataset.X)
                accuracy = accuracy_score(val_dataset.y, preds)

                node_diversity = ensemble.node_diversity()

                result_dict = {
                    'name': ensemble.name,
                    'dataset_name': train_dataset.name,
                    'accuracy': accuracy,
                    'node_diversity': node_diversity,
                }

                self.add_result(**result_dict)
                self.ensembles.append(ensemble)
