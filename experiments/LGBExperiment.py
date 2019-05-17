import lightgbm as lgb

from typing import List
from .Experiment import Experiment
from structure import LGBEnsemble, Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid


class LGBExperiment(Experiment):
    def __init__(self, ):
        super().__init__()

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

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:
                final_params = {**params,
                                'objective': self.pick_objective(train)}

                ensemble = LGBEnsemble(final_params)
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
