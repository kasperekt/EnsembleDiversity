import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Dict
from data import Dataset
from collections import namedtuple

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class Experiment(metaclass=ABCMeta):
    def __init__(self):
        self.results: List[any] = []
        self.EnsembleType: callable = None
        self.param_grid: ParameterGrid = None
        self.name = 'Experiment'

    def reset(self):
        self.results = []

    def add_result(self, **kwargs):
        result = {**kwargs}
        self.results.append(result)

    def to_csv(self, filepath: str):
        if len(self.results) == 0:
            raise ValueError(
                'Results are empty. Try "run" before saving to csv.')

        df = pd.DataFrame(self.results)
        df.to_csv(filepath)

    def get_final_params(self, params: dict, train: Dataset, val: Dataset) -> dict:
        return {**params}

    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        if self.EnsembleType is None:
            raise ValueError('Ensemble type is not specified')

        if self.param_grid is None:
            raise ValueError('Param grid is not defined')

        print(f'Running {self.name} experiment...')

        for train, val in zip(train_data, val_data):
            for params in self.param_grid:  # pylint: disable=not-an-iterable
                final_params = self.get_final_params(params, train, val)

                print(f'[{train.name}] Params: {final_params}')

                ensemble = self.EnsembleType(  # pylint: disable=not-callable
                    final_params)
                ensemble.fit(train)

                preds = ensemble.predict(val)
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
